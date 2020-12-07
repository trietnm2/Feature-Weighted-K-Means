package org.apache.spark.ml.clustering

import org.apache.spark.SparkException
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

trait FeatureWeightingKMeansModel extends KMeansParams {
  protected var trainingSummary: Option[KMeansSummary] = None

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setK(value: Int): this.type = set(k, value)

  def setInitMode(value: String): this.type = set(initMode, value)

  def setInitSteps(value: Int): this.type = set(initSteps, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setTol(value: Double): this.type = set(tol, value)

  def setSeed(value: Long): this.type = set(seed, value)

  /**
   * Return true if there exists summary of model.
   */
  def hasSummary: Boolean = trainingSummary.nonEmpty

  /**
   * Gets summary of model on training set. An exception is
   * thrown if `trainingSummary == None`.
   */
  def summary: KMeansSummary = trainingSummary.getOrElse {
    throw new SparkException(s"No training summary available for the ${this.getClass.getSimpleName}")
  }

  def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  private[clustering] def setSummary(summary: Option[KMeansSummary]): this.type = {
    this.trainingSummary = summary
    this
  }

  def transform(dataset: Dataset[_]): DataFrame = ???

  def computeSilhouetteScore(points: DataFrame, featureCol: String, predictionCol: String): Double = {
    val evaluator: ClusteringEvaluator = new ClusteringEvaluator()
      .setFeaturesCol(featureCol)
      .setPredictionCol(predictionCol)

    val silhouette: Double = evaluator.evaluate(points)
    points.unpersist()
    silhouette
  }
}
