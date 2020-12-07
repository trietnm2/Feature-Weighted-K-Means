package org.apache.spark.mllib.clustering

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.internal.Logging
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
import scala.math.{abs => mAbs, pow => mPow}

/**
 * Triet Nguyen
 * Intelligent Minkowski Weighted K-Means (iMWK-Means)
 */
class IntelligentMinkowskiWeightedKMeans private(k: Int,
                                                 maxIterations: Int,
                                                 initializationMode: String,
                                                 initializationSteps: Int,
                                                 epsilon: Double,
                                                 seed: Long,
                                                 private var p: Double)
  extends FeatureWeightingKMeans(k, maxIterations, initializationMode, initializationSteps, epsilon, seed)
    with Logging {

  def this() = this(2, 20, KMeans.K_MEANS_PARALLEL, 2, 1e-4, Utils.random.nextLong(), 1 + Utils.random.nextDouble())

  def getP: Double = p

  def setP(p: Double): this.type = {
    this.p = p
    this
  }

  def setInitialModel(model: IntelligentMinkowskiWeightedKMeansModel): this.type = {
    require(model.k == k, "mismatched cluster count")
    initialModel = Some(model)
    this
  }

  override def fit(data: RDD[Vector]): IntelligentMinkowskiWeightedKMeansModel = {
    fit(data, None)
  }

  def fit(data: RDD[Vector],
          instr: Option[Instrumentation]): IntelligentMinkowskiWeightedKMeansModel = {

    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning(
        "The input data is not directly cached, which may hurt performance if its"
          + " parent RDDs are also uncached."
      )
    }

    val zippedData = data.map { v =>
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
  private def runAlgorithm(
                            data: RDD[VectorWithNorm],
                            instr: Option[Instrumentation]
                          ): IntelligentMinkowskiWeightedKMeansModel = {

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

      val clusters = getClusters(data, bcCenters.value, bcWeights.value, p)
      updateCenters(clusters, bcCenters.value)
      updateWeights(clusters, bcCenters.value, bcWeights.value)

      bcCenters.destroy(blocking = false)
      bcWeights.destroy(blocking = false)

      val newOptVal = getOptimizationFunctionValue(clusters, centers, weights)
      val cost = (newOptVal - optVal) * (newOptVal - optVal)
      logInfo(s"The opt is $cost.")

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

    new IntelligentMinkowskiWeightedKMeansModel(centers.map(_.vector), weights.map(_.vector), p)
  }

  /**
   * Fixes C and W to minimises Objective Function in respect to S
   *
   * @param data
   * @param centers
   * @param weights
   * @return
   */
  def getClusters(data: RDD[VectorWithNorm],
                  centers: Array[VectorWithNorm],
                  weights: Array[VectorWithNorm],
                  p: Double): RDD[(Int, VectorWithNorm, Double)] = {

    // Find the sum and count of points mapping to each center
    val clusters = data.mapPartitions { points =>
      points.map { point =>
        val (bestCenter, bestDistance) = IntelligentMinkowskiWeightedKMeans.findClosest(centers, point, weights, p)
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

    //Calculate SUM_i_Sk |y_iv - c_kv|^p
    val clusterData = data
      .mapPartitions { points =>
        points.map { point =>
          val bestCenter = point._1
          val center = centers(bestCenter)
          val sumVec = Vectors.dense(
            point._2.vector.toArray
              .zip(center.vector.toArray)
              .map {
                case (y_iv: Double, c_kv: Double) =>
                  mPow(mAbs(y_iv - c_kv), p)
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

    for (i <- clusterData.keys) {
      val weight = weights(i).vector.toArray
      val D_kp = clusterData.get(i).get.toArray
      var newWeightArray = Array.fill(weight.size)(0.0)

      D_kp.zipWithIndex.foreach {
        case (d_kvp: Double, v: Int) =>
          if (d_kvp == 0) {
            newWeightArray(v) = 0.0
          } else {
            val d_kvps = Array.fill(D_kp.length)(d_kvp)

            // Calculate w_v = 1 / (SUM_j_h [(Dv/Dj)^(1/(beta-1))])
            newWeightArray(v) = 1.0 / d_kvps
              .zip(D_kp)
              .map {
                case (dd_kvp: Double, dd_kup: Double) =>
                  //Calculate (Dv/Dj)^(1/(beta-1))
                  mPow(dd_kvp / dd_kup, 1.0 / (p - 1.0))
              }
              .sum
          }
      }

      newWeightArray = newWeightArray.map(x => (if (x.isNaN || x.isInfinity) 0.0 else x))
      val sumWeights = newWeightArray.sum
      if (sumWeights == 0) {
        weights(i) = new VectorWithNorm(Vectors.dense((0 until dims).map(_ => (1.0 / dims)).toArray))
      } else {
        weights(i) = new VectorWithNorm(Vectors.dense(newWeightArray.map(_ / sumWeights)))
      }
    }
  }

  /**
   * The Objective Function of Attribute Weighting K-Means (AWK)
   * SUM_k_K SUM_i_Sk SUM_v_V [(w_kv )^p * |y_iv, c_kv|^p]
   *
   * @param data
   * @param centers
   * @param weights
   * @return
   */
  override def getOptimizationFunctionValue(data: RDD[(Int, VectorWithNorm, Double)],
                                            centers: Array[VectorWithNorm],
                                            weights: Array[VectorWithNorm]): Double = {

    val opt = data
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
                mPow(w_kv * mAbs(y_iv - c_kv), p)
            }
            .sum
          sum
        }
      }
      .reduce {
        case (sum1, sum2) =>
          (sum1 + sum2)
      }

    opt
  }

}

/**
 * Top-level methods for calling K-means clustering.
 */
object IntelligentMinkowskiWeightedKMeans {

  def train(data: RDD[Vector],
            k: Int,
            maxIterations: Int,
            initializationMode: String,
            seed: Long): IntelligentMinkowskiWeightedKMeansModel = {
    new IntelligentMinkowskiWeightedKMeans()
      .setK(k)
      .setMaxIterations(maxIterations)
      .setInitializationMode(initializationMode)
      .setSeed(seed)
      .fit(data)
  }

  def train(data: RDD[Vector], k: Int, maxIterations: Int, initializationMode: String): IntelligentMinkowskiWeightedKMeansModel = {
    new IntelligentMinkowskiWeightedKMeans()
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
            seed: Long): IntelligentMinkowskiWeightedKMeansModel = {
    new IntelligentMinkowskiWeightedKMeans()
      .setK(k)
      .setMaxIterations(maxIterations)
      .setInitializationMode(initializationMode)
      .setSeed(seed)
      .fit(data)
  }

  def train(data: RDD[Vector],
            k: Int,
            maxIterations: Int,
            runs: Int,
            initializationMode: String): IntelligentMinkowskiWeightedKMeansModel = {
    new IntelligentMinkowskiWeightedKMeans()
      .setK(k)
      .setMaxIterations(maxIterations)
      .setInitializationMode(initializationMode)
      .fit(data)
  }

  def train(data: RDD[Vector], k: Int, maxIterations: Int): IntelligentMinkowskiWeightedKMeansModel = {
    new IntelligentMinkowskiWeightedKMeans()
      .setK(k)
      .setMaxIterations(maxIterations)
      .fit(data)
  }

  def train(data: RDD[Vector], k: Int, maxIterations: Int, runs: Int): IntelligentMinkowskiWeightedKMeansModel = {
    new IntelligentMinkowskiWeightedKMeans()
      .setK(k)
      .setMaxIterations(maxIterations)
      .fit(data)
  }

  private[mllib] def pointCost(centers: TraversableOnce[VectorWithNorm],
                               point: VectorWithNorm,
                               weights: TraversableOnce[VectorWithNorm],
                               p: Double): Double =
    findClosest(centers, point, weights, p)._2

  def calDistance(diffVec: Vector, weight: Vector, p: Double): Double = {
    val distance: Double = diffVec.toArray
      .zip(weight.toArray)
      .map {
        case (diff: Double, w_kv: Double) =>
          // diff = (y_iv - c_kv)
          mPow(w_kv * mAbs(diff), p)
      }
      .sum
    distance
  }

  /**
   * The assignment of entity to the closest cluster Sk uses the weighted distance
   * d(y_i, c_k) = SUM_v_V [ w_kv^p * |y_iv - c_kv|^p ]
   *
   * @param centers
   * @param point
   * @param weights
   * @return
   */
  def findClosest(centers: TraversableOnce[VectorWithNorm],
                  point: VectorWithNorm,
                  weights: TraversableOnce[VectorWithNorm],
                  p: Double): (Int, Double) = {

    var bestDistance = Double.PositiveInfinity
    var bestIndex = 0
    var i = 0

    val centers_weights = centers.toArray.zip(weights.toIterable)
    centers_weights.foreach { cw =>
      val diffVec: Vector = Vectors.zeros(point.vector.size)
      copy(point.vector, diffVec)
      axpy(-1.0, cw._1.vector, diffVec)

      // Calculate d(y_i, c_k) = SUM_v_V [w_kv * (y_iv - c_kv)^2]
      val distance: Double = calDistance(diffVec, cw._2.vector, p)
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
                        p: Double): Double = {

    var outerDissimilarity: Double = Double.PositiveInfinity
    var innerDissimilarity: Double = Double.PositiveInfinity

    val centers_weights = centers.toArray.zip(weights.toIterable)
    centers_weights.foreach { cw =>
      val diffVec: Vector = Vectors.zeros(point.vector.size)
      copy(point.vector, diffVec)
      axpy(-1.0, cw._1.vector, diffVec)

      // Calculate d(y_i, c_k) = SUM_v_V [w_kv * (y_iv - c_kv)^2]
      val distance: Double = calDistance(diffVec, cw._2.vector, p)
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

class IntelligentMinkowskiWeightedKMeansModel(clusterCenters: Array[Vector], clusterWeights: Array[Vector], val p: Double)
  extends FeatureWeightingKMeansModel(clusterCenters, clusterWeights) {

  def this(centers: java.lang.Iterable[Vector], weights: java.lang.Iterable[Vector], p: Double) =
    this(centers.asScala.toArray, weights.asScala.toArray, p)

  override def predict(point: Vector): Int = {
    IntelligentMinkowskiWeightedKMeans.findClosest(clusterCentersWithNorm, new VectorWithNorm(point), clusterWeightsWithNorm, p)._1
  }

  override def predict(points: RDD[Vector]): RDD[Int] = {
    val centersWithNorm = clusterCentersWithNorm
    val weightsWithNorm = clusterWeightsWithNorm
    val bcCentersWithNorm = points.context.broadcast(centersWithNorm)
    val bcWeightsWithNorm = points.context.broadcast(weightsWithNorm)
    points.map(
      x =>
        IntelligentMinkowskiWeightedKMeans.findClosest(bcCentersWithNorm.value, new VectorWithNorm(x), bcWeightsWithNorm.value, p)._1
    )
  }

  override def computeCost(data: RDD[Vector]): Double = {
    val centersWithNorm = clusterCentersWithNorm
    val weightsWithNorm = clusterWeightsWithNorm
    val bcCentersWithNorm = data.context.broadcast(centersWithNorm)
    val bcWeightsWithNorm = data.context.broadcast(weightsWithNorm)
    data
      .map(
        x => IntelligentMinkowskiWeightedKMeans.pointCost(bcCentersWithNorm.value, new VectorWithNorm(x), bcWeightsWithNorm.value, p)
      )
      .sum()
  }

  def computeSilhouette(data: RDD[Vector]): Double = {
    val centersWithNorm = clusterCentersWithNorm
    val weightsWithNorm = clusterWeightsWithNorm
    val bcCentersWithNorm = data.context.broadcast(centersWithNorm)
    val bcWeightsWithNorm = data.context.broadcast(weightsWithNorm)
    data.map(x =>
      IntelligentMinkowskiWeightedKMeans.computeSilhouette(bcCentersWithNorm.value, new VectorWithNorm(x), bcWeightsWithNorm.value, p)
    ).mean()
  }

  override def save(sc: SparkContext, path: String): Unit = {
    IntelligentMinkowskiWeightedKMeansModel.SaveLoadV1_0.save(sc, this, path)
  }
}

object IntelligentMinkowskiWeightedKMeansModel extends Loader[IntelligentMinkowskiWeightedKMeansModel] {

  override def load(sc: SparkContext, path: String): IntelligentMinkowskiWeightedKMeansModel = {
    IntelligentMinkowskiWeightedKMeansModel.SaveLoadV1_0.load(sc, path)
  }

  private case class Cluster(id: Int, point: Vector)

  object SaveLoadV1_0 {

    private val thisFormatVersion = "1.0"

    private[clustering] val thisClassName = "org.apache.spark.mllib.clustering.IntelligentMinkowskiWeightedKMeansModel"

    def save(sc: SparkContext, model: IntelligentMinkowskiWeightedKMeansModel, path: String): Unit = {
      val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
      val metadata = compact(
        render(("class" -> thisClassName) ~ ("version" -> thisFormatVersion) ~ ("k" -> model.k) ~ ("p" -> model.p))
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

    def load(sc: SparkContext, path: String): IntelligentMinkowskiWeightedKMeansModel = {
      implicit val formats = DefaultFormats
      val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
      val (className, formatVersion, metadata) = Loader.loadMetadata(sc, path)
      assert(className == thisClassName)
      assert(formatVersion == thisFormatVersion)
      val k = (metadata \ "k").extract[Int]
      val p = (metadata \ "p").extract[Double]
      val centroids = spark.read.parquet(centersPath(path))
      Loader.checkSchema[Cluster](centroids.schema)
      val localCentroids = centroids.rdd.map(Cluster.apply).collect()
      assert(k == localCentroids.length)

      val weights = spark.read.parquet(weightsPath(path))
      Loader.checkSchema[Cluster](weights.schema)
      val localWeights = weights.rdd.map(Cluster.apply).collect()
      assert(k == localWeights.length)

      new IntelligentMinkowskiWeightedKMeansModel(localCentroids.sortBy(_.id).map(_.point), localWeights.sortBy(_.id).map(_.point), p)
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
