

import lakala.tools.cleanData.CleanData
import lakala.tools.cleanData.GBTGenerateData
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics


/**
  * Created by dyh on 2016/7/22.
  */
object GbtLr {
  def main(args: Array[String]) {

    LogManager.getRootLogger.setLevel(Level.ERROR)

    val sparkConf = new SparkConf().setAppName("brca")
    val sc = new SparkContext(sparkConf)

    val  brcaRow = sc.textFile("/data/mllib/brca/brca.fpkm.txt",5)

    val kv = sc.textFile("/data/mllib/brca/sample_label.txt")

    val afterR2C = CleanData.row2col(brcaRow)

    val dataLabeledPoint = CleanData.label(afterR2C, kv)

    val transformedData = GBTGenerateData.transformData(dataLabeledPoint)


    val (trainData, testData) = transformedData
    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(trainData)
    val predAndLabel = testData.map{ lp =>
      (model.predict(lp.features), lp.label)
    }
    val metrics = new MulticlassMetrics(predAndLabel)
    val precision = metrics.precision
    println("Precision = " + precision)
  }
}
