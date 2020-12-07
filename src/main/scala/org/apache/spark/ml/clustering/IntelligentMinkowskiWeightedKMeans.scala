package org.apache.spark.ml.clustering

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.linalg.{Vector, VectorUDT}
import org.apache.spark.ml.param.{DoubleParam, ParamMap, ParamValidators}
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.clustering.{IntelligentMinkowskiWeightedKMeans => MLlibIntelligentMinkowskiWeightedKMeans, IntelligentMinkowskiWeightedKMeansModel => MLlibIntelligentMinkowskiWeightedKMeansModel, KMeans => MLlibKMeans}
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.mllib.linalg.{Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.util.Utils
import org.apache.spark.util.VersionUtils.majorVersion
import org.json4s.jackson.JsonMethods._

private[clustering] trait IntelligentMinkowskiWeightedKMeansParams extends FeatureWeightingKMeansModel {
  final val p = new DoubleParam(
    this,
    "p",
    "The hyperparameter. " +
      "Must be > 1.",
    ParamValidators.gt(1)
  )

  def setP(value: Double): this.type = set(p, value)

  def getP: Double = $(p)
}

class IntelligentMinkowskiWeightedKMeansModel private[ml](val uid: String,
                                                          private val parentModel: MLlibIntelligentMinkowskiWeightedKMeansModel)
  extends Model[IntelligentMinkowskiWeightedKMeansModel]
    with IntelligentMinkowskiWeightedKMeansParams
    with MLWritable {

  def copy(extra: ParamMap): IntelligentMinkowskiWeightedKMeansModel = {
    val copied = copyValues(new IntelligentMinkowskiWeightedKMeansModel(uid, parentModel), extra)
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

  override def write: MLWriter = new IntelligentMinkowskiWeightedKMeansModel.IntelligentMinkowskiWeightedKMeansModelWriter(this)
}

object IntelligentMinkowskiWeightedKMeansModel extends MLReadable[IntelligentMinkowskiWeightedKMeansModel] {
  override def read: MLReader[IntelligentMinkowskiWeightedKMeansModel] = new IntelligentMinkowskiWeightedKMeansModelReader

  override def load(path: String): IntelligentMinkowskiWeightedKMeansModel = super.load(path)

  /** Helper class for storing model data */
  private case class Data(clusterIdx: Int, clusterCenter: Vector)

  private case class OldData(clusterCenters: Array[OldVector])

  private class IntelligentMinkowskiWeightedKMeansModelWriter(instance: IntelligentMinkowskiWeightedKMeansModel) extends MLWriter {

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

  private class IntelligentMinkowskiWeightedKMeansModelReader extends MLReader[IntelligentMinkowskiWeightedKMeansModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[IntelligentMinkowskiWeightedKMeansModel].getName

    override def load(path: String): IntelligentMinkowskiWeightedKMeansModel = {
      // Import implicits for Dataset Encoder
      val sparkSession = super.sparkSession
      import sparkSession.implicits._

      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val p = compact(render(metadata.getParamValue("p"))).toDouble

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

      val model =
        new IntelligentMinkowskiWeightedKMeansModel(
          metadata.uid,
          new MLlibIntelligentMinkowskiWeightedKMeansModel(clusterCenters, clusterWeights, p)
        )
      metadata.getAndSetParams(model)
      model
    }
  }

}

class IntelligentMinkowskiWeightedKMeans(val uid: String)
  extends Estimator[IntelligentMinkowskiWeightedKMeansModel]
    with IntelligentMinkowskiWeightedKMeansParams
    with DefaultParamsWritable {

  setDefault(
    k -> 2,
    maxIter -> 20,
    initMode -> MLlibKMeans.K_MEANS_PARALLEL,
    initSteps -> 2,
    tol -> 1e-4,
    p -> (1 + Utils.random.nextDouble())
  )

  override def copy(extra: ParamMap): IntelligentMinkowskiWeightedKMeans = defaultCopy(extra)

  def this() = this(Identifiable.randomUID("IntelligentMinkowskiWeightedKMeans"))

  def fit(dataset: Dataset[_]): IntelligentMinkowskiWeightedKMeansModel = instrumented { instr =>
    transformSchema(dataset.schema, logging = true)
    val rdd: RDD[OldVector] = dataset.select(col($(featuresCol))).rdd.map {
      case Row(point: Vector) => OldVectors.fromML(point)
    }

    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, featuresCol, predictionCol, k, initMode, initSteps, maxIter, seed, tol, p)

    val algo = new MLlibIntelligentMinkowskiWeightedKMeans()
      .setK($(k))
      .setInitializationMode($(initMode))
      .setInitializationSteps($(initSteps))
      .setMaxIterations($(maxIter))
      .setSeed($(seed))
      .setEpsilon($(tol))
      .setP($(p))
    val parentModel = algo.fit(rdd, Option(instr))
    val model = copyValues(new IntelligentMinkowskiWeightedKMeansModel(uid, parentModel).setParent(this))
    val summary = new KMeansSummary(model.transform(dataset), $(predictionCol), $(featuresCol), $(k),
      parentModel.numIter, parentModel.trainingCost)
    model.setSummary(Some(summary))
    instr.logSuccess()
    model
  }
}
