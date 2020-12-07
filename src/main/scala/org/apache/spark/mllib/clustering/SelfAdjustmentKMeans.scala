package org.apache.spark.mllib.clustering

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.internal.Logging
import org.apache.spark.ml.clustering.{SelfAdjustmentKMeans => NewSelfAdjustmentKMeans}
import org.apache.spark.ml.util.Instrumentation
import org.apache.spark.mllib.linalg.BLAS.{axpy, copy, scal}
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
import scala.math.{pow => mPow}

/**
 * Triet Nguyen
 * Feature Weight Self-Adjustment K-Means (FWSA)
 */
class SelfAdjustmentKMeans private(k: Int,
                                   maxIterations: Int,
                                   initializationMode: String,
                                   initializationSteps: Int,
                                   epsilon: Double,
                                   seed: Long)
  extends FeatureWeightingKMeans(k, maxIterations, initializationMode, initializationSteps, epsilon, seed)
    with Logging {

  def this() = this(2, 20, KMeans.K_MEANS_PARALLEL, 2, 1e-4, Utils.random.nextLong())

  def setInitialModel(model: SelfAdjustmentKMeansModel): this.type = {
    require(model.k == k, "mismatched cluster count")
    initialModel = Some(model)
    this
  }

  override def fit(data: RDD[Vector]): SelfAdjustmentKMeansModel = {
    fit(data, None)
  }

  def fit(data: RDD[Vector], instr: Option[Instrumentation]): SelfAdjustmentKMeansModel = {

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

  override protected def initCentersAndWeights(data: RDD[VectorWithNorm]): (Array[VectorWithNorm], Array[VectorWithNorm]) = {
    val (centers, _) = super.initCentersAndWeights(data)
    val dims = centers.head.vector.size
    val weights = Array.fill(1)(new VectorWithNorm(Vectors.dense((0 until dims).map(_ => (1.0 / dims)).toArray)))
    (centers, weights)
  }

  /**
   * Implementation of K-Means algorithm.
   */
  private def runAlgorithm(data: RDD[VectorWithNorm],
                           instr: Option[Instrumentation]): SelfAdjustmentKMeansModel = {

    val initStartTime = System.nanoTime()
    val (centers, weights) = initCentersAndWeights(data)
    val initTimeInSeconds = (System.nanoTime() - initStartTime) / 1e9
    logInfo(f"Initialization with $initializationMode took $initTimeInSeconds%.3f seconds.")

    var converged = false
    var iteration = 0
    val optVal = 0
    val iterationStartTime = System.nanoTime()

    instr.foreach(_.logNumFeatures(centers.head.vector.size))
    val centerV = data
      .map(_.vector)
      .reduce {
        case (vec1, vec2) =>
          axpy(1.0, vec2, vec1)
          vec1
      }

    // Execute iterations of Lloyd's algorithm until converged
    while (iteration < maxIterations && !converged) {
      val sc = data.sparkContext
      val bcCenters = sc.broadcast(centers)
      val bcWeights = sc.broadcast(weights)
      val bcCenterV = sc.broadcast(centerV)

      val clusters = getClusters(data, bcCenters.value, bcWeights.value)
      updateCenters(clusters, bcCenters.value)
      updateWeights(clusters, bcCenters.value, bcWeights.value, bcCenterV.value)

      bcCenters.destroy(blocking = false)
      bcWeights.destroy(blocking = false)
      bcCenterV.destroy(blocking = false)

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

    new SelfAdjustmentKMeansModel(centers.map(_.vector), weights.map(_.vector))
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
        val (bestCenter, bestDistance) = SelfAdjustmentKMeans.findClosest(centers, point, weights.head)
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
  def updateWeights(data: RDD[(Int, VectorWithNorm, Double)],
                    centers: Array[VectorWithNorm],
                    weights: Array[VectorWithNorm],
                    centerV: Vector): Unit = {
    val dims = centers.head.vector.size
    val distanceWeight = Vectors.dense(Array.fill(dims)(1.0))

    // Calculate vector a_v = SUM_k_K SUM_i_Sk d(y_iv,c_kv)
    val vec_a = data
      .mapPartitions { points =>
        points.map {
          case (bestCluster, point, _) =>
            SelfAdjustmentKMeans.getDistanceVector(point.vector, centers(bestCluster).vector, distanceWeight)
        }
      }
      .reduce {
        case (vec1, vec2) =>
          axpy(1.0, vec2, vec1)
          vec1
      }

    val map_Nk = data
      .map { case (bestCenter, point, bestDistance) => (bestCenter, 1) }
      .reduceByKey { (count1, count2) =>
        count1 + count2
      }
      .sortBy(_._1)
      .collect()

    // Calculate vector b_v = SUM_k_K N_k d(c_kv,c_v)
    val vec_b = centers
      .zip(map_Nk)
      .map {
        case (c_k, (_, n_k)) =>
          val vec: Vector = SelfAdjustmentKMeans.getDistanceVector(c_k.vector, centerV, distanceWeight)
          scal(n_k, vec)
          vec
      }
      .reduce { (vec1: Vector, vec2: Vector) =>
        axpy(1.0, vec2, vec1)
        vec1
      }

    // Calculate vector w_v = 0.5[w_v + (b_v/a_v)/(SUM_v_V b_v/a_v)]
    val vec_ba = vec_a.toArray.zip(vec_b.toArray).map {
      case (a_v, b_v) =>
        if (a_v == 0.0) 0.0 else b_v / a_v
    }
    val sum_vec_ba = vec_ba.sum
    var newWeightArray = weights.head.vector.toArray
      .zip(vec_ba)
      .map {
        case (w_v: Double, ba_v: Double) =>
          val ba = if (sum_vec_ba == 0) 0.0 else (ba_v / sum_vec_ba)
          0.5 * (w_v + ba)
      }

    newWeightArray = newWeightArray.map(x => (if (x.isNaN || x.isInfinity) 0.0 else x))
    val sumWeights = newWeightArray.sum
    if (sumWeights == 0) {
      weights(0) = new VectorWithNorm(Vectors.dense((0 until dims).map(_ => (1.0 / dims)).toArray))
    } else {
      weights(0) = new VectorWithNorm(Vectors.dense(newWeightArray.map(_ / sumWeights)))
    }
  }

  /**
   * The Objective Function of Attribute Weighting K-Means (AWK)
   * SUM_k_K SUM_i_Sk SUM_v_V [w_v * (y_iv - c_kv)^2]
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

    val opt = data
      .mapPartitions { points =>
        points.map { point =>
          val bestCenter = point._1
          val center = centers(bestCenter)
          val weight = weights.head
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

    opt
  }

}

/**
 * Top-level methods for calling K-means clustering.
 */
object SelfAdjustmentKMeans {

  def train(data: RDD[Vector], k: Int, maxIterations: Int, initializationMode: String, seed: Long): SelfAdjustmentKMeansModel = {
    new SelfAdjustmentKMeans()
      .setK(k)
      .setMaxIterations(maxIterations)
      .setInitializationMode(initializationMode)
      .setSeed(seed)
      .fit(data)
  }

  def train(data: RDD[Vector], k: Int, maxIterations: Int, initializationMode: String): SelfAdjustmentKMeansModel = {
    new SelfAdjustmentKMeans()
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
            seed: Long): SelfAdjustmentKMeansModel = {
    new SelfAdjustmentKMeans()
      .setK(k)
      .setMaxIterations(maxIterations)
      .setInitializationMode(initializationMode)
      .setSeed(seed)
      .fit(data)
  }

  def train(data: RDD[Vector], k: Int, maxIterations: Int, runs: Int, initializationMode: String): SelfAdjustmentKMeansModel = {
    new SelfAdjustmentKMeans()
      .setK(k)
      .setMaxIterations(maxIterations)
      .setInitializationMode(initializationMode)
      .fit(data)
  }

  def train(data: RDD[Vector], k: Int, maxIterations: Int): SelfAdjustmentKMeansModel = {
    new SelfAdjustmentKMeans()
      .setK(k)
      .setMaxIterations(maxIterations)
      .fit(data)
  }

  def train(data: RDD[Vector], k: Int, maxIterations: Int, runs: Int): SelfAdjustmentKMeansModel = {
    new SelfAdjustmentKMeans()
      .setK(k)
      .setMaxIterations(maxIterations)
      .fit(data)
  }

  private[mllib] def pointCost(centers: TraversableOnce[VectorWithNorm], point: VectorWithNorm, weights: VectorWithNorm): Double =
    findClosest(centers, point, weights)._2

  /** *
   *
   * @param v1
   * @param v2
   * @param weights
   * @return
   */
  def getDistanceVector(v1: Vector, v2: Vector, weights: Vector): Vector = {
    val sum: Vector = Vectors.zeros(v1.size)
    copy(v1, sum)
    axpy(-1.0, v2, sum)

    // Calculate d(y_i, c_k) = SUM_v_V [w_v * (y_iv - c_kv)^2]
    Vectors.dense(
      sum.toArray
        .zip(weights.toArray)
        .map {
          case (diff: Double, w_v: Double) =>
            // diff = (y_iv - c_kv)
            w_v * diff * diff
        }
    )
  }

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
  def findClosest(centers: TraversableOnce[VectorWithNorm], point: VectorWithNorm, weights: VectorWithNorm): (Int, Double) = {

    var bestDistance = Double.PositiveInfinity
    var bestIndex = 0
    var i = 0

    centers.foreach { c =>
      val distance: Double = getDistanceVector(point.vector, c.vector, weights.vector).toArray.sum

      if (distance < bestDistance) {
        bestDistance = distance
        bestIndex = i
      }
      i += 1
    }
    (bestIndex, bestDistance)
  }

  def computeSilhouette(centers: TraversableOnce[VectorWithNorm], point: VectorWithNorm, weights: VectorWithNorm): Double = {

    var outerDissimilarity: Double = Double.PositiveInfinity
    var innerDissimilarity: Double = Double.PositiveInfinity

    centers.foreach { c =>
      val distance: Double = getDistanceVector(point.vector, c.vector, weights.vector).toArray.sum

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

class SelfAdjustmentKMeansModel(clusterCenters: Array[Vector], clusterWeights: Array[Vector])
  extends FeatureWeightingKMeansModel(clusterCenters, clusterWeights) {

  def this(centers: java.lang.Iterable[Vector], weights: java.lang.Iterable[Vector]) =
    this(centers.asScala.toArray, weights.asScala.toArray)

  override def predict(point: Vector): Int = {
    SelfAdjustmentKMeans.findClosest(clusterCentersWithNorm, new VectorWithNorm(point), clusterWeightsWithNorm.head)._1
  }

  override def predict(points: RDD[Vector]): RDD[Int] = {
    val centersWithNorm = clusterCentersWithNorm
    val weightsWithNorm = clusterWeightsWithNorm
    val bcCentersWithNorm = points.context.broadcast(centersWithNorm)
    val bcWeightsWithNorm = points.context.broadcast(weightsWithNorm)
    points.map(p => SelfAdjustmentKMeans.findClosest(bcCentersWithNorm.value, new VectorWithNorm(p), bcWeightsWithNorm.value.head)._1)
  }

  override def computeCost(data: RDD[Vector]): Double = {
    val centersWithNorm = clusterCentersWithNorm
    val weightsWithNorm = clusterWeightsWithNorm
    val bcCentersWithNorm = data.context.broadcast(centersWithNorm)
    val bcWeightsWithNorm = data.context.broadcast(weightsWithNorm)
    data.map(p =>
      SelfAdjustmentKMeans.pointCost(bcCentersWithNorm.value, new VectorWithNorm(p), bcWeightsWithNorm.value.head)
    ).sum()
  }

  def computeSilhouette(data: RDD[Vector]): Double = {
    val centersWithNorm = clusterCentersWithNorm
    val weightsWithNorm = clusterWeightsWithNorm
    val bcCentersWithNorm = data.context.broadcast(centersWithNorm)
    val bcWeightsWithNorm = data.context.broadcast(weightsWithNorm)
    data.map { p =>
      SelfAdjustmentKMeans.computeSilhouette(bcCentersWithNorm.value, new VectorWithNorm(p), bcWeightsWithNorm.value.head)
    }.mean()
  }

  override def save(sc: SparkContext, path: String): Unit = {
    SelfAdjustmentKMeansModel.SaveLoadV1_0.save(sc, this, path)
  }
}

object SelfAdjustmentKMeansModel extends Loader[SelfAdjustmentKMeansModel] {

  override def load(sc: SparkContext, path: String): SelfAdjustmentKMeansModel = {
    SelfAdjustmentKMeansModel.SaveLoadV1_0.load(sc, path)
  }

  private case class Cluster(id: Int, point: Vector)

  object SaveLoadV1_0 {

    private val thisFormatVersion = "1.0"

    private[clustering] val thisClassName = "org.apache.spark.mllib.clustering.SelfAdjustmentKMeansModel"

    def save(sc: SparkContext, model: SelfAdjustmentKMeansModel, path: String): Unit = {
      val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
      val metadata = compact(render(("class" -> thisClassName) ~ ("version" -> thisFormatVersion) ~ ("k" -> model.k)))
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

    def weightsPath(path: String): String = new Path(path, "weights").toUri.toString

    def centersPath(path: String): String = new Path(path, "centers").toUri.toString

    def load(sc: SparkContext, path: String): SelfAdjustmentKMeansModel = {
      implicit val formats = DefaultFormats
      val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
      val (className, formatVersion, metadata) = Loader.loadMetadata(sc, path)
      assert(className == thisClassName)
      assert(formatVersion == thisFormatVersion)
      val k = (metadata \ "k").extract[Int]
      val centroids = spark.read.parquet(centersPath(path))
      Loader.checkSchema[Cluster](centroids.schema)
      val localCentroids = centroids.rdd.map(Cluster.apply).collect()
      assert(k == localCentroids.length)

      val weights = spark.read.parquet(weightsPath(path))
      Loader.checkSchema[Cluster](weights.schema)
      val localWeights = weights.rdd.map(Cluster.apply).collect()
      assert(k == localWeights.length)

      new SelfAdjustmentKMeansModel(localCentroids.sortBy(_.id).map(_.point), localWeights.sortBy(_.id).map(_.point))
    }
  }

  private object Cluster {

    def apply(r: Row): Cluster = {
      Cluster(r.getInt(0), r.getAs[Vector](1))
    }
  }

}
