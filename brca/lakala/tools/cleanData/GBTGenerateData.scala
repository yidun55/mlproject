package lakala.tools.cleanData

import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.{BoostingStrategy, FeatureType, Strategy}
import org.apache.spark.mllib.tree.model.Node
import org.apache.spark.rdd.RDD

/**
  * Created by dyh on 2016/7/28.
  */
class GBTGenerateData {

  private def getLeafNodes(node:Node):Array[Int] = {

    var treeLeafNodes = new Array[Int](0)
    if(node.isLeaf){
      treeLeafNodes = treeLeafNodes :+ node.id
    }else{
      treeLeafNodes = treeLeafNodes ++ getLeafNodes(node.leftNode.get)
      treeLeafNodes = treeLeafNodes ++ getLeafNodes(node.rightNode.get)
    }
    treeLeafNodes
  }

  private def predictModify(node:Node, features:DenseVector):Int = {

    val split = node.split
    if(node.isLeaf){
      node.id
    }else{
      if(split.get.featureType==FeatureType.Continuous){
        if(features(split.get.feature) <= split.get.threshold){
          predictModify(node.leftNode.get, features)
        }else{
          predictModify(node.rightNode.get, features)
        }
      }else{
        if(split.get.categories.contains(features(split.get.feature))){
          predictModify(node.leftNode.get, features)
        }else{
          predictModify(node.rightNode.get, features)
        }
      }
    }
  }
}


object GBTGenerateData{

  val gbtGenerateData = new GBTGenerateData()

  def transformData(rdd:RDD[LabeledPoint]):(RDD[LabeledPoint], RDD[LabeledPoint] )= {

    val numTrees = 20

    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.setNumIterations(numTrees)      //设置树的多少
    val treeStrategy = Strategy.defaultStrategy("Classification")
    treeStrategy.setNumClasses(2)
    treeStrategy.setMaxDepth(30)
//    treeStrategy.setCategoricalFeaturesInfo(Map[Int, Int]())
    boostingStrategy.setTreeStrategy(treeStrategy)

    val splits = rdd.randomSplit(Array(0.7, 0.3))
    val (trainData, testData) = (splits(0), splits(1))
    val model = GradientBoostedTrees.train(trainData, boostingStrategy)

    val labelAndPred = testData.map{ lp =>
      (lp.label, model.predict(lp.features))
    }
    val error = labelAndPred.filter(lp=>lp._1!=lp._2).count.toDouble / labelAndPred.count



    val treeLeafArray = new Array[Array[Int]](numTrees)
    for(i <- 0 until numTrees){
      treeLeafArray(i) = gbtGenerateData.getLeafNodes(model.trees(i).topNode)
    }

    val newFeatureData = (data:RDD[LabeledPoint]) => data.map{ lp =>      //匿名函数用于转换特征
      var newFeature = new Array[Double](0)
      for(i <- 0 until numTrees){
        val nodeId = gbtGenerateData.predictModify(model.trees(i).topNode, lp.features.toDense)
        val leafArray = new Array[Double]((model.trees(i).numNodes+1)/2)
        leafArray(treeLeafArray(i).indexOf(nodeId)) = 1
        newFeature = newFeature ++ leafArray
      }
      LabeledPoint(lp.label, new DenseVector(newFeature))
    }

    val newTrainData = newFeatureData(trainData)
    val newTestData = newFeatureData(testData)

    (newTrainData, newTestData)
  }

}