package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator





object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/

val df = spark.read.parquet("../data/prepared_trainingset")


    /** TF-IDF **/

val tokenizer = new RegexTokenizer()
.setPattern("\\W+")
.setGaps(true)
.setInputCol("text")
.setOutputCol("tokens")

val remover = new StopWordsRemover()
.setInputCol("tokens")
.setOutputCol("filtered")

val cvModel = new CountVectorizer()
.setInputCol("filtered")
.setOutputCol("TF")

val idf = new IDF()
  .setInputCol("TF")
  .setOutputCol("tfidf")


val countryIndexer = new StringIndexer()
  .setInputCol("country2")
  .setOutputCol("country_indexed")


val currencyIndexer = new StringIndexer()
  .setInputCol("currency2")
  .setOutputCol("currency_indexed")





    /** VECTOR ASSEMBLER **/




val assembler = new VectorAssembler()
  .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"))
  .setOutputCol("features")


    /** MODEL **/

val lr = new LogisticRegression()
  .setElasticNetParam(0.0)
  .setFitIntercept(true)
  .setFeaturesCol("features")
  .setLabelCol("final_status")
  .setStandardization(true)
  .setPredictionCol("predictions")
  .setRawPredictionCol("raw_predictions")
  .setThresholds(Array(0.7, 0.3))
  .setTol(1.0e-6)
  .setMaxIter(300)



    /** PIPELINE **/



val pipeline = new Pipeline()
  .setStages(Array(tokenizer, remover, cvModel, idf, countryIndexer, currencyIndexer, assembler, lr))





    /** TRAINING AND GRID-SEARCH **/


val Array(training, test) = df.randomSplit(Array(0.9,0.1))

val paramGrid = new ParamGridBuilder()
  .addGrid(cvModel.minDF,(55.0 to 95.0 by 20))
  .addGrid(cvModel.minDF,Array(10e-8, 10e-6, 10e-4, 10e-2))
  .build()

val eval = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(eval)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

val model = trainValidationSplit.fit(training)

val df_WithPredictions = model.transform(test)
df_WithPredictions.select("features","predictions").show()
df_WithPredictions.groupBy("final_status", "predictions").count.show()








  }
}
