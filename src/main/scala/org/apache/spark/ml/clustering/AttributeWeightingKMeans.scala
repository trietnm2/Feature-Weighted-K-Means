package org.apache.spark.ml.clustering

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.linalg.{Vector, VectorUDT}
import org.apache.spark.ml.param.{DoubleParam, ParamMap, ParamValidators}
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.clustering.{AttributeWeightingKMeans => MLlibAttributeWeightingKMeans, AttributeWeightingKMeansModel => MLlibAttributeWeightingKMeansModel, KMeans => MLlibKMeans}
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.mllib.linalg.{Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.util.Utils
import org.apache.spark.util.VersionUtils.majorVersion
import org.json4s.jackson.JsonMethods._

private[clustering] trait AttributeWeightingKMeansParams extends FeatureWeightingKMeansModel {
  final val beta = new DoubleParam(
    this,
    "beta",
    "The hyperparameter. " +
      "Must be > 1.",
    ParamValidators.gt(1)
  )

  def setBeta(value: Double): this.type = set(beta, value)

  def getBeta: Double = $(beta)
}

class AttributeWeightingKMeansModel private[ml](val uid: String, private val parentModel: MLlibAttributeWeightingKMeansModel)
  extends Model[AttributeWeightingKMeansModel]
    with AttributeWeightingKMeansParams
    with MLWritable {

  def copy(extra: ParamMap): AttributeWeightingKMeansModel = {
    val copied = copyValues(new AttributeWeightingKMeansModel(uid, parentModel), extra)
    copied.setSummary(trainingSummary).setParent(this.parent)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val predictUDF = udf((vector: Vector) => predict(vector))
    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }

  def predict(features: Vector): Int = parentModel.predict(features)

  def clusterCenters: Array[Vector] = parentModel.clusterCenters.map(_.asML)

  def clusterWeights: Array[Vector] = parentModel.clusterWeights.map(_.asML)

  /**
   * Return the K-means cost (sum of squared distances of points to their nearest center) for this
   * model on the given data.
   */
  def computeCost(dataset: Dataset[_]): Double = {
    SchemaUtils.checkColumnType(dataset.schema, $(featuresCol), new VectorUDT)
    val data: RDD[OldVector] = dataset.select(col($(featuresCol))).rdd.map {
      case Row(point: Vector) => OldVectors.fromML(point)
    }
    parentModel.computeCost(data)
  }

  def computeSilhouette(dataset: Dataset[_]): Double = {
    SchemaUtils.checkColumnType(dataset.schema, $(featuresCol), new VectorUDT)
    val data: RDD[OldVector] = dataset.select(col($(featuresCol))).rdd.map {
      case Row(point: Vector) => OldVectors.fromML(point)
    }
    parentModel.computeSilhouette(data)
  }

  override def write: MLWriter = new AttributeWeightingKMeansModel.AttributeWeightingKMeansModelWriter(this)
}

object AttributeWeightingKMeansModel extends MLReadable[AttributeWeightingKMeansModel] {
  override def read: MLReader[AttributeWeightingKMeansModel] = new AttributeWeightingKMeansModelReader

  override def load(path: String): AttributeWeightingKMeansModel = super.load(path)

  /** Helper class for storing model data */
  private case class Data(clusterIdx: Int, clusterCenter: Vector)

  private case class OldData(clusterCenters: Array[OldVector])

  private class AttributeWeightingKMeansModelWriter(instance: AttributeWeightingKMeansModel) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      // Save metadata and Params
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      // Save model data: cluster centers
      val centers: Array[Data] = instance.clusterCenters.zipWithIndex.map {
        case (center, idx) =>
          Data(idx, center)
      }
      val centersPath = new Path(path, "centers").toString
      sparkSession.createDataFrame(centers).repartition(1).write.parquet(centersPath)
      // Save model data: cluster weights
      val weights: Array[Data] = instance.clusterWeights.zipWithIndex.map {
        case (center, idx) =>
          Data(idx, center)
      }
      val weightsPath = new Path(path, "weights").toString
      sparkSession.createDataFrame(weights).repartition(1).write.parquet(weightsPath)
    }
  }

  private class AttributeWeightingKMeansModelReader extends MLReader[AttributeWeightingKMeansModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[AttributeWeightingKMeansModel].getName

    override def load(path: String): AttributeWeightingKMeansModel = {
      // Import implicits for Dataset Encoder
      val sparkSession = super.sparkSession
      import sparkSession.implicits._

      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val beta = compact(render(metadata.getParamValue("beta"))).toDouble

      val centersPath = new Path(path, "centers").toString
      val clusterCenters = if (majorVersion(metadata.sparkVersion) >= 2) {
        val centers: Dataset[Data] = sparkSession.read.parquet(centersPath).as[Data]
        centers.collect().sortBy(_.clusterIdx).map(_.clusterCenter).map(OldVectors.fromML)
      } else {
        // Loads KMeansModel stored with the old format used by Spark 1.6 and earlier.
        sparkSession.read.parquet(centersPath).as[OldData].head().clusterCenters
      }

      val weightsPath = new Path(path, "weights").toString
      val clusterWeights = if (majorVersion(metadata.sparkVersion) >= 2) {
        val weights: Dataset[Data] = sparkSession.read.parquet(weightsPath).as[Data]
        weights.collect().sortBy(_.clusterIdx).map(_.clusterCenter).map(OldVectors.fromML)
      } else {
        // Loads KMeansModel stored with the old format used by Spark 1.6 and earlier.
        sparkSession.read.parquet(weightsPath).as[OldData].head().clusterCenters
      }

      val model = new AttributeWeightingKMeansModel(metadata.uid,
        new MLlibAttributeWeightingKMeansModel(clusterCenters, clusterWeights, beta))
      metadata.getAndSetParams(model)
      model
    }
  }

}

class AttributeWeightingKMeans(val uid: String)
  extends Estimator[AttributeWeightingKMeansModel]
    with AttributeWeightingKMeansParams
    with DefaultParamsWritable {

  setDefault(
    k -> 2,
    maxIter -> 20,
    initMode -> MLlibKMeans.K_MEANS_PARALLEL,
    initSteps -> 2,
    tol -> 1e-4,
    beta -> (1 + Utils.random.nextDouble())
  )

  override def copy(extra: ParamMap): AttributeWeightingKMeans = defaultCopy(extra)

  def this() = this(Identifiable.randomUID("AttributeWeightingKMeans"))

  def fit(dataset: Dataset[_]): AttributeWeightingKMeansModel = instrumented { instr =>
    transformSchema(dataset.schema, logging = true)
    val rdd: RDD[OldVector] = dataset.select(col($(featuresCol))).rdd.map {
      case Row(point: Vector) => OldVectors.fromML(point)
    }

    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, featuresCol, predictionCol, k, initMode, initSteps, maxIter, seed, tol, beta)

    val algo = new MLlibAttributeWeightingKMeans()
      .setK($(k))
      .setInitializationMode($(initMode))
      .setInitializationSteps($(initSteps))
      .setMaxIterations($(maxIter))
      .setSeed($(seed))
      .setEpsilon($(tol))
      .setBeta($(beta))
    val parentModel = algo.fit(rdd, Option(instr))
    val model = copyValues(new AttributeWeightingKMeansModel(uid, parentModel).setParent(this))
    val summary = new KMeansSummary(model.transform(dataset), $(predictionCol), $(featuresCol), $(k),
      parentModel.numIter, parentModel.trainingCost)
    model.setSummary(Some(summary))
    instr.logSuccess()
    model
  }
}
