package org.apache.spark.mllib.clustering

import org.apache.spark.mllib.linalg.BLAS.{axpy, copy, scal}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.XORShiftRandom

/**
  * Triet Nguyen
  * Feature Weighting K-Means
  */
abstract class FeatureWeightingKMeans(protected var k:                   Int,
                                      protected var maxIterations:       Int,
                                      protected var initializationMode:  String,
                                      protected var initializationSteps: Int,
                                      protected var epsilon:             Double,
                                      protected var seed:                Long)
    extends KMeans {
  protected var initialModel: Option[FeatureWeightingKMeansModel] = None

  def fit(data: RDD[Vector]): FeatureWeightingKMeansModel = ???

  protected def initCentersAndWeights(data: RDD[VectorWithNorm]): (Array[VectorWithNorm], Array[VectorWithNorm]) = {
    val centers = initialModel match {
      case Some(kMeansCenters) =>
        kMeansCenters.clusterCenters.map(new VectorWithNorm(_))
      case None =>
        if (initializationMode == KMeans.RANDOM) {
          initRandom(data)
        } else {
          val distanceMeasureInstance = DistanceMeasure.decodeFromString(DistanceMeasure.EUCLIDEAN)
          initKMeansParallel(data, distanceMeasureInstance)
        }
    }
    val dims    = centers.head.vector.size
    val weights = Array.fill(centers.length)(new VectorWithNorm(Vectors.dense((0 until dims).map(_ => (1.0 / dims)).toArray)))

    (centers, weights)
  }

  /**
    * Initialize a set of cluster centers at random.
    */
  def initRandom(data: RDD[VectorWithNorm]): Array[VectorWithNorm] = {
    // Select without replacement; may still produce duplicates if the data has < k distinct
    // points, so deduplicate the centroids to match the behavior of k-means|| in the same situation
    data
      .takeSample(false, k, new XORShiftRandom(this.seed).nextInt())
      .map(_.vector)
      .distinct
      .map(new VectorWithNorm(_))
  }

  /**
    * Fixes C and W to minimises Objective Function in respect to S
    *
    * @param data
    * @param centers
    * @param weights
    * @return
    */
  protected def getClusters(data:    RDD[VectorWithNorm],
                            centers: Array[VectorWithNorm],
                            weights: Array[VectorWithNorm]): RDD[(Int, VectorWithNorm, Double)] = ???

  /**
    * Fixes S and W to minimises Objective Function in respect to C
    *
    * @param data
    * @param centers
    */
  protected def updateCenters(data: RDD[(Int, VectorWithNorm, Double)], centers: Array[VectorWithNorm]): Unit = {
    val dims = centers.head.vector.size

    // Find the sum and count of points mapping to each center
    // c_kv = (1/|Sk|) * SUM_i_Sk [y_iv]
    val totalContribs = data
      .mapPartitions { points =>
        val sums   = Array.fill(centers.length)(Vectors.zeros(dims))
        val counts = Array.fill(centers.length)(0L)

        points.foreach { point =>
          val bestCenter = point._1
          val sum        = sums(bestCenter)
          axpy(1.0, point._2.vector, sum)
          counts(bestCenter) += 1
        }

        counts.indices.filter(counts(_) > 0).map(j => (j, (sums(j), counts(j)))).iterator
      }
      .reduceByKey {
        case ((sum1, count1), (sum2, count2)) =>
          axpy(1.0, sum2, sum1)
          (sum1, count1 + count2)
      }
      .collectAsMap()

    // Update the cluster centers and costs
    totalContribs.foreach {
      case (j, (sum, count)) =>
        scal(1.0 / count, sum)
        centers(j) = new VectorWithNorm(sum)
    }
  }

  /**
    * Fixes S and C to minimises Objective Function in respect to W
    *
    * @param data
    * @param centers
    * @param weights
    */
  protected def updateWeights(data:    RDD[(Int, VectorWithNorm, Double)],
                              centers: Array[VectorWithNorm],
                              weights: Array[VectorWithNorm]): Unit =
    ???

  /**
    * The Objective Function of Feature Weighting K-Means
    *
    * @param data
    * @param centers
    * @param weights
    * @return
    */
  protected def getOptimizationFunctionValue(data:    RDD[(Int, VectorWithNorm, Double)],
                                             centers: Array[VectorWithNorm],
                                             weights: Array[VectorWithNorm]): Double = ???
}

object FeatureWeightingKMeans {

  /**
    * Map ((A, B), C)) to (A,B,C)
    *
    * @param t
    * @tparam A
    * @tparam B
    * @tparam C
    * @return
    */
  def f2[A, B, C](t: ((A, B), C)) = (t._1._1, t._1._2, t._2)

  def calDistance(diffVec: Vector, weight: Vector): Double ={
    val distance: Double = diffVec.toArray
      .zip(weight.toArray)
      .map {
        case (diff: Double, w_kv: Double) =>
          // diff = (y_iv - c_kv)
          w_kv * diff * diff
      }
      .sum
    distance
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
  def findClosest(centers: TraversableOnce[VectorWithNorm],
                  point:   VectorWithNorm,
                  weights: TraversableOnce[VectorWithNorm]): (Int, Double) = {

    var bestDistance = Double.PositiveInfinity
    var bestIndex    = 0
    var i            = 0

    val centers_weights = centers.toArray.zip(weights.toIterable)
    centers_weights.foreach { cw =>
      val diffVec: Vector = Vectors.zeros(point.vector.size)
      copy(point.vector, diffVec)
      axpy(-1.0, cw._1.vector, diffVec)

      // Calculate d(y_i, c_k) = SUM_v_V [w_kv * (y_iv - c_kv)^2]
      val distance: Double =  calDistance(diffVec, cw._2.vector)
      if (distance < bestDistance) {
        bestDistance = distance
        bestIndex    = i
      }
      i += 1
    }
    (bestIndex, bestDistance)
  }

  def calSilhouetteCoefficient(centers: TraversableOnce[VectorWithNorm],
                               point:   VectorWithNorm,
                               weights: TraversableOnce[VectorWithNorm]
                              ): Double = {

    var outerDissimilarity:Double = Double.PositiveInfinity
    var innerDissimilarity:Double = Double.PositiveInfinity

    val centers_weights = centers.toArray.zip(weights.toIterable)
    centers_weights.foreach { cw =>
      val diffVec: Vector = Vectors.zeros(point.vector.size)
      copy(point.vector, diffVec)
      axpy(-1.0, cw._1.vector, diffVec)

      // Calculate d(y_i, c_k) = SUM_v_V [w_kv * (y_iv - c_kv)^2]
      val distance: Double =  calDistance(diffVec, cw._2.vector)
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

abstract class FeatureWeightingKMeansModel(clusterCenters: Array[Vector], val clusterWeights: Array[Vector])
    extends KMeansModel(clusterCenters) {

  protected def clusterCentersWithNorm: Iterable[VectorWithNorm] =
    clusterCenters.map(new VectorWithNorm(_))

  protected def clusterWeightsWithNorm: Iterable[VectorWithNorm] =
    clusterWeights.map(new VectorWithNorm(_))
}
