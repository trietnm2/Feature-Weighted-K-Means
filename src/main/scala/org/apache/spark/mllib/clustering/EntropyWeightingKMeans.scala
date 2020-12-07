package org.apache.spark.mllib.clustering

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.internal.Logging
import org.apache.spark.ml.clustering.{EntropyWeightingKMeans => NewEntropyWeightingKMeans}
import org.apache.spark.ml.util.Instrumentation
import org.apache.spark.mllib.linalg.BLAS.{axpy, copy}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.Loader
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{compact, render}
import org.json4s.{DefaultFormats, _}

import scala.collection.JavaConverters._
import scala.math.{exp => mExp, log => mLog}

/**
 * Triet Nguyen
 * Entropy Weighting K-Means (EW-KM)
 */
class EntropyWeightingKMeans private(k: Int,
                                     maxIterations: Int,
                                     initializationMode: String,
                                     initializationSteps: Int,
                                     epsilon: Double,
                                     seed: Long,
                                     private var gamma: Double)
  extends FeatureWeightingKMeans(k, maxIterations, initializationMode, initializationSteps, epsilon, seed)
    with Logging {

  def this() = this(2, 20, KMeans.K_MEANS_PARALLEL, 2, 1e-4, Utils.random.nextLong(), Utils.random.nextDouble())

  def getGamma: Double = gamma

  def setGamma(gamma: Double): this.type = {
    this.gamma = gamma
    this
  }

  def setInitialModel(model: EntropyWeightingKMeansModel): this.type = {
    require(model.k == k, "mismatched cluster count")
    initialModel = Some(model)
    this
  }

  override def fit(data: RDD[Vector]): EntropyWeightingKMeansModel = {
    fit(data, None)
  }

  def fit(data: RDD[Vector], instr: Option[Instrumentation]): EntropyWeightingKMeansModel = {

    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning(
        "The input data is not directly cached, which may hurt performance if its"
          + " parent RDDs are also uncached."
      )
    }

    // Compute squared norms and cache them.
    val zippedData = data.map {
      case (v) =>
        new VectorWithNorm(v, 0.0)
    }
    val model = runAlgorithm(zippedData, instr)

    // Warn at the end of the run as well, for increased visibility.
    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning(
        "The input data was not directly cached, which may hurt performance if its"
          + " parent RDDs are also uncached."
      )
    }
    model
  }

  /**
   * Implementation of K-Means algorithm.
   */
  private def runAlgorithm(data: RDD[VectorWithNorm],
                           instr: Option[Instrumentation]): EntropyWeightingKMeansModel = {

    val initStartTime = System.nanoTime()
    val (centers, weights) = initCentersAndWeights(data)
    val initTimeInSeconds = (System.nanoTime() - initStartTime) / 1e9
    logInfo(f"Initialization with $initializationMode took $initTimeInSeconds%.3f seconds.")

    var converged = false
    var iteration = 0
    val optVal = 0
    val iterationStartTime = System.nanoTime()

    instr.foreach(_.logNumFeatures(centers.head.vector.size))

    // Execute iterations of Lloyd's algorithm until converged
    while (iteration < maxIterations && !converged) {
      val sc = data.sparkContext
      val bcCenters = sc.broadcast(centers)
      val bcWeights = sc.broadcast(weights)

      val clusters = getClusters(data, bcCenters.value, bcWeights.value)
      updateCenters(clusters, bcCenters.value)
      updateWeights(clusters, bcCenters.value, bcWeights.value)

      bcCenters.destroy(blocking = false)
      bcWeights.destroy(blocking = false)

      val newOptVal = getOptimizationFunctionValue(clusters, centers, weights)
      val cost = (newOptVal - optVal) * (newOptVal - optVal)
      logInfo(s"The cost is $cost.")

      converged = true
      if (cost > epsilon * epsilon) {
        converged = false
      }
      iteration += 1
    }

    val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
    logInfo(f"Iterations took $iterationTimeInSeconds%.3f seconds.")

    if (iteration == maxIterations) {
      logInfo(s"KMeans reached the max number of iterations: $maxIterations.")
    } else {
      logInfo(s"KMeans converged in $iteration iterations.")
    }

    new EntropyWeightingKMeansModel(centers.map(_.vector), weights.map(_.vector), gamma)
  }

  /**
   * Fixes C and W to minimises Objective Function in respect to S
   *
   * @param data
   * @param centers
   * @param weights
   * @return
   */
  override def getClusters(data: RDD[VectorWithNorm],
                           centers: Array[VectorWithNorm],
                           weights: Array[VectorWithNorm]): RDD[(Int, VectorWithNorm, Double)] = {

    // Find the sum and count of points mapping to each center
    val clusters = data.mapPartitions { points =>
      points.map { point =>
        val (bestCenter, bestDistance) = EntropyWeightingKMeans.findClosest(centers, point, weights, gamma)
        (bestCenter, point, bestDistance)
      }
    }

    clusters
  }

  /**
   * Fixes S and C to minimises Objective Function in respect to W
   *
   * @param data
   * @param centers
   * @param weights
   */
  override def updateWeights(data: RDD[(Int, VectorWithNorm, Double)],
                             centers: Array[VectorWithNorm],
                             weights: Array[VectorWithNorm]): Unit = {
    val dims = centers.head.vector.size

    //Calculate D_kv = (y_iv - c_kv)^2
    val D_k = data
      .mapPartitions { points =>
        points.map { point =>
          val bestCenter = point._1
          val center = centers(bestCenter)
          val sumVec = Vectors.dense(
            point._2.vector.toArray
              .zip(center.vector.toArray)
              .map {
                case (y_iv: Double, c_kv: Double) =>
                  (y_iv - c_kv) * (y_iv - c_kv)
              }
          )
          (bestCenter, sumVec)
        }
      }
      .reduceByKey {
        case (sum1, sum2) =>
          axpy(1.0, sum1, sum2)
          sum2
      }
      .collectAsMap()

    for (i <- D_k.keys) {
      val numerator = D_k.get(i).get.toArray.map(D_kv => mExp(-mLog(D_kv) / gamma))
      var denominator = numerator.sum
      denominator = if (denominator == 0) 1.0 else denominator
      val newWeight = numerator
        .zip(Array.fill(numerator.length)(denominator))
        .map {
          case (numerator_kv: Double, denominator_kv: Double) =>
            // numerator_kv = exp(-D_kv/gamma)
            // denominator_kv = SUM_j_V exp(-D_kj/gamma)
            val value = numerator_kv / denominator_kv
            (if (value.isInfinity || value.isNaN) 0.0 else value)
        }

      val sumWeights = newWeight.sum
      if (sumWeights == 0) {
        weights(i) = new VectorWithNorm(Vectors.dense((0 until dims).map(_ => (1.0 / dims)).toArray))
      } else {
        weights(i) = new VectorWithNorm(Vectors.dense(newWeight.map(_ / sumWeights)))
      }

    }
  }

  /**
   * The Objective Function of Entropy Weighting K-Means (AWK)
   * SUM_k_K [SUM_i_Sk SUM_v_V [(w_kv )*d(y_iv, c_kv)] + gamma * SUM_v_V [w_kv*log(w_kv)] ]
   * where d(y_iv, c_kv) = (y_iv - c_kv)^2
   *
   *
   * @param data
   * @param centers
   * @param weights
   * @return
   */
  override def getOptimizationFunctionValue(data: RDD[(Int, VectorWithNorm, Double)],
                                            centers: Array[VectorWithNorm],
                                            weights: Array[VectorWithNorm]): Double = {

    // Calculate SUM_k_K SUM_i_Sk SUM_v_V [w_kv * (y_iv - c_kv)^2]
    val w01 = data
      .mapPartitions { points =>
        points.map { point =>
          val bestCenter = point._1
          val center = centers(bestCenter)
          val weight = weights(bestCenter)
          val sum = point._2.vector.toArray
            .zip(center.vector.toArray)
            .zip(weight.vector.toArray)
            .map(FeatureWeightingKMeans.f2(_))
            .map {
              case (y_iv: Double, c_kv: Double, w_kv: Double) =>
                w_kv * (y_iv - c_kv) * (y_iv - c_kv)
            }
            .sum
          sum
        }
      }
      .reduce {
        case (sum1, sum2) =>
          (sum1 + sum2)
      }

    // Calculate SUM_k_K [gamma * SUM_v_V (w_kv * log(w_kv))]]
    val w02 = weights.map { wk =>
      wk.vector.toArray.map { w_kv =>
        if (w_kv > 0) w_kv * mLog(w_kv) else 0.0
      }.sum * gamma
    }.sum

    val opt = w01 + w02
    opt
  }

}

/**
 * Top-level methods for calling K-means clustering.
 */
object EntropyWeightingKMeans {

  def train(data: RDD[Vector], k: Int, maxIterations: Int, initializationMode: String, seed: Long): EntropyWeightingKMeansModel = {
    new EntropyWeightingKMeans()
      .setK(k)
      .setMaxIterations(maxIterations)
      .setInitializationMode(initializationMode)
      .setSeed(seed)
      .fit(data)
  }

  def train(data: RDD[Vector], k: Int, maxIterations: Int, initializationMode: String): EntropyWeightingKMeansModel = {
    new EntropyWeightingKMeans()
      .setK(k)
      .setMaxIterations(maxIterations)
      .setInitializationMode(initializationMode)
      .fit(data)
  }

  def train(data: RDD[Vector],
            k: Int,
            maxIterations: Int,
            runs: Int,
            initializationMode: String,
            seed: Long): EntropyWeightingKMeansModel = {
    new EntropyWeightingKMeans()
      .setK(k)
      .setMaxIterations(maxIterations)
      .setInitializationMode(initializationMode)
      .setSeed(seed)
      .fit(data)
  }

  def train(data: RDD[Vector], k: Int, maxIterations: Int, runs: Int, initializationMode: String): EntropyWeightingKMeansModel = {
    new EntropyWeightingKMeans()
      .setK(k)
      .setMaxIterations(maxIterations)
      .setInitializationMode(initializationMode)
      .fit(data)
  }

  def train(data: RDD[Vector], k: Int, maxIterations: Int): EntropyWeightingKMeansModel = {
    new EntropyWeightingKMeans()
      .setK(k)
      .setMaxIterations(maxIterations)
      .fit(data)
  }

  def train(data: RDD[Vector], k: Int, maxIterations: Int, runs: Int): EntropyWeightingKMeansModel = {
    new EntropyWeightingKMeans()
      .setK(k)
      .setMaxIterations(maxIterations)
      .fit(data)
  }

  private[mllib] def pointCost(centers: TraversableOnce[VectorWithNorm],
                               point: VectorWithNorm,
                               weights: TraversableOnce[VectorWithNorm],
                               gamma: Double
                              ): Double =
    findClosest(centers, point, weights, gamma)._2

  /**
   * The assignment of entity to the closest cluster Sk uses the weighted distance
   * d(y_i, c_k) = SUM_v_V [ w_kv * (y_iv - c_kv)^2 ]
   *
   *
   * @param centers
   * @param point
   * @param weights
   * @return
   */

  def findClosest(centers: TraversableOnce[VectorWithNorm],
                  point: VectorWithNorm,
                  weights: TraversableOnce[VectorWithNorm],
                  gamma: Double
                 ): (Int, Double) = {

    var bestDistance = Double.PositiveInfinity
    var bestIndex = 0
    var i = 0

    val centers_weights = centers.toArray.zip(weights.toIterable)
    centers_weights.foreach { cw =>
      val diffVec: Vector = Vectors.zeros(point.vector.size)
      copy(point.vector, diffVec)
      axpy(-1.0, cw._1.vector, diffVec)

      // Calculate d(y_i, c_k) = SUM_v_V [w_kv * (y_iv - c_kv)^2]
      val d1: Double = FeatureWeightingKMeans.calDistance(diffVec, cw._2.vector)
      val d2: Double = gamma * cw._2.vector.toArray.map(w_kv => w_kv * mLog(w_kv)).sum
      val distance: Double = d1 + d2

      if (distance < bestDistance) {
        bestDistance = distance
        bestIndex = i
      }
      i += 1
    }
    (bestIndex, bestDistance)
  }

  def computeSilhouette(centers: TraversableOnce[VectorWithNorm],
                        point: VectorWithNorm,
                        weights: TraversableOnce[VectorWithNorm],
                        gamma: Double
                       ): Double = {

    var outerDissimilarity: Double = Double.PositiveInfinity
    var innerDissimilarity: Double = Double.PositiveInfinity

    val centers_weights = centers.toArray.zip(weights.toIterable)
    centers_weights.foreach { cw =>
      val diffVec: Vector = Vectors.zeros(point.vector.size)
      copy(point.vector, diffVec)
      axpy(-1.0, cw._1.vector, diffVec)

      // Calculate d(y_i, c_k) = SUM_v_V [w_kv * (y_iv - c_kv)^2]
      val d1: Double = FeatureWeightingKMeans.calDistance(diffVec, cw._2.vector)
      val d2: Double = gamma * cw._2.vector.toArray.map(w_kv => w_kv * mLog(w_kv)).sum
      val distance: Double = d1 + d2

      if (distance < innerDissimilarity) {
        outerDissimilarity = innerDissimilarity
        innerDissimilarity = distance
      }
    }

    if (innerDissimilarity < outerDissimilarity) {
      1 - (innerDissimilarity / outerDissimilarity)
    } else if (innerDissimilarity > outerDissimilarity) {
      (outerDissimilarity / innerDissimilarity) - 1
    } else {
      0.0
    }
  }
}

class EntropyWeightingKMeansModel(clusterCenters: Array[Vector], clusterWeights: Array[Vector], val gamma: Double)
  extends FeatureWeightingKMeansModel(clusterCenters, clusterWeights) {
  def this(centers: java.lang.Iterable[Vector], weights: java.lang.Iterable[Vector], gamma: Double) =
    this(centers.asScala.toArray, weights.asScala.toArray, gamma)

  override def predict(point: Vector): Int = {
    EntropyWeightingKMeans.findClosest(clusterCentersWithNorm, new VectorWithNorm(point), clusterWeightsWithNorm, gamma)._1
  }

  override def predict(points: RDD[Vector]): RDD[Int] = {
    val centersWithNorm = clusterCentersWithNorm
    val weightsWithNorm = clusterWeightsWithNorm
    val bcCentersWithNorm = points.context.broadcast(centersWithNorm)
    val bcWeightsWithNorm = points.context.broadcast(weightsWithNorm)
    points.map(p => EntropyWeightingKMeans.findClosest(bcCentersWithNorm.value, new VectorWithNorm(p), bcWeightsWithNorm.value, gamma)._1)
  }

  override def computeCost(data: RDD[Vector]): Double = {
    val centersWithNorm = clusterCentersWithNorm
    val weightsWithNorm = clusterWeightsWithNorm
    val bcCentersWithNorm = data.context.broadcast(centersWithNorm)
    val bcWeightsWithNorm = data.context.broadcast(weightsWithNorm)
    data.map(p =>
      EntropyWeightingKMeans.pointCost(bcCentersWithNorm.value, new VectorWithNorm(p), bcWeightsWithNorm.value, gamma)
    ).sum()
  }

  def computeSilhouette(data: RDD[Vector]): Double = {
    val centersWithNorm = clusterCentersWithNorm
    val weightsWithNorm = clusterWeightsWithNorm
    val bcCentersWithNorm = data.context.broadcast(centersWithNorm)
    val bcWeightsWithNorm = data.context.broadcast(weightsWithNorm)
    data.map(p =>
      EntropyWeightingKMeans.computeSilhouette(bcCentersWithNorm.value, new VectorWithNorm(p), bcWeightsWithNorm.value, gamma)
    ).mean()
  }

  override def save(sc: SparkContext, path: String): Unit = {
    EntropyWeightingKMeansModel.SaveLoadV1_0.save(sc, this, path)
  }
}

object EntropyWeightingKMeansModel extends Loader[EntropyWeightingKMeansModel] {

  override def load(sc: SparkContext, path: String): EntropyWeightingKMeansModel = {
    EntropyWeightingKMeansModel.SaveLoadV1_0.load(sc, path)
  }

  private case class Cluster(id: Int, point: Vector)

  object SaveLoadV1_0 {

    private val thisFormatVersion = "1.0"

    private[clustering] val thisClassName = "org.apache.spark.mllib.clustering.EntropyWeightingKMeansModel"

    def save(sc: SparkContext, model: EntropyWeightingKMeansModel, path: String): Unit = {
      val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
      val metadata = compact(
        render(("class" -> thisClassName) ~ ("version" -> thisFormatVersion) ~ ("k" -> model.k) ~ ("gamma" -> model.gamma))
      )
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(Loader.metadataPath(path))
      val centerRDD = sc.parallelize(model.clusterCenters.zipWithIndex).map {
        case (point, id) =>
          Cluster(id, point)
      }
      spark.createDataFrame(centerRDD).write.parquet(centersPath(path))
      val weightRDD = sc.parallelize(model.clusterWeights.zipWithIndex).map {
        case (point, id) =>
          Cluster(id, point)
      }
      spark.createDataFrame(weightRDD).write.parquet(weightsPath(path))
    }

    def load(sc: SparkContext, path: String): EntropyWeightingKMeansModel = {
      implicit val formats = DefaultFormats
      val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
      val (className, formatVersion, metadata) = Loader.loadMetadata(sc, path)
      assert(className == thisClassName)
      assert(formatVersion == thisFormatVersion)
      val k = (metadata \ "k").extract[Int]
      val gamma = (metadata \ "gamma").extract[Double]
      val centroids = spark.read.parquet(centersPath(path))
      Loader.checkSchema[Cluster](centroids.schema)
      val localCentroids = centroids.rdd.map(Cluster.apply).collect()
      assert(k == localCentroids.length)

      val weights = spark.read.parquet(weightsPath(path))
      Loader.checkSchema[Cluster](weights.schema)
      val localWeights = weights.rdd.map(Cluster.apply).collect()
      assert(k == localWeights.length)

      new EntropyWeightingKMeansModel(localCentroids.sortBy(_.id).map(_.point), localWeights.sortBy(_.id).map(_.point), gamma)
    }

    def weightsPath(path: String): String = new Path(path, "weights").toUri.toString

    def centersPath(path: String): String = new Path(path, "centers").toUri.toString
  }

  private object Cluster {

    def apply(r: Row): Cluster = {
      Cluster(r.getInt(0), r.getAs[Vector](1))
    }
  }

}
