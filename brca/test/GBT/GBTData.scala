package test.GBT



import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.Node
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.tree.configuration.FeatureType

/**
  * Created by dyh on 2016/7/28.
  */
class GBTData {

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

object GBTData{


}