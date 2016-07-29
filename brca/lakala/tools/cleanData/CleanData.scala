package lakala.tools.cleanData


import org.apache.spark.rdd.RDD
import scala.collection.mutable.{ArrayBuffer, Map}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.log4j.{ConsoleAppender, Logger}
import org.apache.spark.{SparkConf, SparkContext}


/**
  * Created by dyh on 2016/7/20.
  * 行转列
  */
class CleanData {

  val logger = Logger.getRootLogger
  val appender = new ConsoleAppender()


  private def partitionRow2Col(rdd:RDD[String]):RDD[(Int,ArrayBuffer[Any])]={

    val rddR2C = rdd.mapPartitionsWithIndex{
      (index:Int, iter:Iterator[String]) =>
      var tmpContainer:Map[Int, ArrayBuffer[Any]] = Map()

      while(iter.hasNext){
        val line:String = iter.next().stripLineEnd
        val fields = line.split("\t")
        for(i<- 0 until fields.length){
          if(tmpContainer.contains(i)){
            tmpContainer(i) += fields(i)
          }else{
            tmpContainer += (i -> ArrayBuffer(index,fields(i)))
          }
        }
      }
        tmpContainer.iterator
    }
    rddR2C
  }


  private def combinPartition(rdd: RDD[(Int,ArrayBuffer[Any])]):RDD[Array[Any]] = {

    val groupRdd = rdd.groupByKey().values.map{
      el =>
        val sortedData = el.toList.sortBy(arrBuffer => arrBuffer(0).toString.toDouble)
        val withoutIndexData = sortedData.map{ arrB =>
          arrB.remove(0)
          arrB
        }
        val te = withoutIndexData.flatMap(l=>l.toList)
        te.toArray
    }
    groupRdd
  }

  private def label(rdd1:RDD[Array[Any]], rdd2:RDD[String]):RDD[LabeledPoint] = {

    val nameCol = rdd1.first().indexOf("Gene")
    if(nameCol == -1){
//      logger.error("error in CleanData.label index")   //找不到Gene name
      sys.exit()
    }

    val kv1 = rdd2.map{line =>                         //获得样本和样本label的映射,例如Map("000d877f-8d03-44bc-8607-27b5ba84b5fe","NT")
      val tArr = line.stripLineEnd.split("\t")
      (tArr(0) -> tArr(1))
    }.collect()
    val kv = kv1.toMap

    val kv3 = (rdd:RDD[String]) => {
      val arr = rdd.map{line =>                         //获得样本和样本label的映射,例如Map("000d877f-8d03-44bc-8607-27b5ba84b5fe","NT")
        val tArr = line.stripLineEnd.split("\t")
        (tArr(0) -> tArr(1))
      }.collect()
      arr
    }
    val result = kv3

    val sc:SparkContext = rdd1.sparkContext       //只为测试，可删除
    val broadcastVar =  sc.broadcast(Array(1,2))

    val removedFirstRdd = removeFirstLine(rdd1)    //去掉第一行样本名

    val labelFeature:RDD[Array[Any]] = removedFirstRdd.map{ arr =>
      val arrBuffer = arr.toBuffer
      val key:String = arrBuffer(nameCol).toString
      arrBuffer.remove(nameCol)
      if(kv.contains(key)) {
        val label: Any = if (kv(key) == "NT") 1.0 else 0.0
        arrBuffer.append(label)
      }
      arrBuffer.toArray
    }

    val labeledPointRdd = createLabeledPoint(labelFeature)   //获得LabeledPoint

    labeledPointRdd
  }



  private def createLabeledPoint(rdd:RDD[Array[Any]]):RDD[LabeledPoint] = {

    val lpRdd = rdd.map{ arr1=>
      val arrBuffer = arr1.toBuffer
      val label = arrBuffer.remove(arrBuffer.length-1)
      val arrB:ArrayBuffer[Double] = ArrayBuffer()
      for(el <- arrBuffer){
        try{
          arrB.append(el.toString.toDouble)
        }catch{
          case _:NumberFormatException =>
        }
      }
      val arr:Array[Double] = arrB.toArray
      val lp = LabeledPoint(label.toString.toDouble, Vectors.dense(arr))
      lp
    }
    lpRdd
  }


  /*
  去掉第一行样本名
  * */
  private def removeFirstLine(rdd:RDD[Array[Any]]): RDD[Array[Any]] = {
    val rdd1 = rdd.filter{ arrBuffer =>
      var reBoolean = false
      try{
        arrBuffer(1).toString.toDouble   //除零以外
        reBoolean = true
      }catch {
        case _: NumberFormatException =>
      }
      reBoolean
    }
    rdd1
  }
}


object CleanData{

  def row2col(rdd : RDD[String]): RDD[Array[Any]] ={

    val cleanData = new CleanData()
    val rdd1 = cleanData.partitionRow2Col(rdd)
    val rdd2 = cleanData.combinPartition(rdd1)
    rdd2
  }

  def label(rdd1:RDD[Array[Any]], rdd2:RDD[String]):RDD[LabeledPoint] ={

    val cleanData = new CleanData()
    val result = cleanData.label(rdd1, rdd2)
    result
  }
}