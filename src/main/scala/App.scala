
import org.apache.spark.ml.clustering._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

import java.util.logging.{Level, Logger}

object App {
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local").appName("Test01").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")


    val schema = StructType(
      List(
        StructField("Driver_ID", StringType, true),
        StructField("Distance_Feature", DoubleType, true),
        StructField("Speeding_Feature", DoubleType, true)
      )
    )
    var dataset = spark.read
      .format("csv")
      .option("delimiter", "\t")
      .option("header", "true")
      .schema(schema)
      .load("data/data_1024.csv")
      .drop("Driver_ID")
    //dataset.show()

    val assembler = new VectorAssembler()
      .setInputCols(Array("Distance_Feature", "Speeding_Feature"))
      .setOutputCol("features")

    dataset = assembler.transform(dataset)
    //dataset.show()

    (2 until 3).foreach { k =>
      println("K:", k)
      //val kmeans = new AttributeWeightingKMeans().setK(k).setBeta(3).setSeed(1L).setMaxIter(1000)
      //val model = kmeans.fit(dataset)
      //val modelPath = s"model/AttributeWeightingKMeans_$k"
      //val model = AttributeWeightingKMeansModel.load(modelPath)

      //val kmeans = new WeightedKMeans().setK(k).setBeta(10).setSeed(1L).setMaxIter(100000)
      //val model = kmeans.fit(dataset)
      //val modelPath = s"model/WeightedKMeans_$k"
      //val model = WeightedKMeansModel.load(modelPath)

      //val kmeans = new EntropyWeightingKMeans().setK(k).setGamma(1).setSeed(1L).setMaxIter(100000)
      //val model = kmeans.fit(dataset)
      //val modelPath = s"model/EntropyWeightingKMeans_$k"
      //val model = EntropyWeightingKMeansModel.load(modelPath)

      //val kmeans = new IntelligentMinkowskiWeightedKMeans().setK(k).setP(2).setSeed(1L).setMaxIter(100000)
      //val model = kmeans.fit(dataset)
      //val modelPath = s"model/IntelligentMinkowskiWeightedKMeans_$k"
      //val model = IntelligentMinkowskiWeightedKMeansModel.load(modelPath)

      val kmeans = new SelfAdjustmentKMeans().setK(k).setSeed(1L).setMaxIter(100000)
      val model = kmeans.fit(dataset)
      val modelPath = s"model/SelfAdjustmentKMeans_$k"
      //val model = SelfAdjustmentKMeansModel.load(modelPath)


      val WSSSE = model.computeCost(dataset)
      println(s"Within Set Sum of Squared Errors = $WSSSE")

      // Calculate Silhouette Score for Feature Weighted K-Means
      val silhouette = model.computeSilhouette(dataset)
      println(s"Silhouette Weighted Score = $silhouette")

      // Calculate Silhouette Score for Normal K-Means
      val getCluster = udf { features: Vector => model.predict(features) }
      val dfPredicted = dataset.withColumn("prediction", getCluster(col("features")))
      val silhouette_1 = model.computeSilhouetteScore(dfPredicted, "features", "prediction")
      println(s"Silhouette MlLib Score = $silhouette_1")

      // Shows the result.
      println("Cluster Centers: ")
      model.clusterCenters.foreach(println)

      println("Cluster Weights: ")
      model.clusterWeights.foreach(println)

      model.write.overwrite().save(modelPath)
    }
  }
}
