import jdk.internal.org.objectweb.asm.tree.MultiANewArrayInsnNode

import scala.collection.mutable

/**
  * Created by stevanusandrianta on 5/12/16.
  */

case class Perceptron(perceptronId: String = "perceptron")

case class Layer(layerId: String, perceptron: List[Perceptron]){

  var nextLayer : Layer = null
  var prevLayer : Layer = null

  def setNextLayer(layer: Layer) = {
    this.nextLayer = layer
  }

  def setPreviousLayer(layer: Layer) = {
    this.prevLayer = layer
  }

}

case class Network(networkId: String, inputLayer : Layer, hiddenLayer: List[Layer], outputLayer: Layer){

  var synapses: List[(String, String, Double)] = _

  def addSynapses(from: String, to: String, weight: Double) = {
    synapses :+ (from, to, weight)
  }

}

object NeuralNetwork {

  def initiateNetwork(dataInput: List[List[Double]], learningRate: Double,
                       nInput: Int, nHidden: List[Int], nOutput: Int, maxIteration: Int) = {

    var hiddenLayer = nHidden.map{ item =>
      new Layer("inputLayer", new Array[Perceptron](item).toList)
    }
    var inputLayer = new Layer("input", new Array[Perceptron](nInput).toList)
    var outputLayer = new Layer("output", new Array[Perceptron](nOutput).toList)

    inputLayer.setNextLayer(hiddenLayer.head)
    hiddenLayer.zipWithIndex.foreach{ layer =>
      if(layer._2 == 0){
        layer._1.setNextLayer(hiddenLayer(layer._2 + 1))
      }
      else if(layer._2+1 != hiddenLayer.size){
        layer._1.setNextLayer(hiddenLayer(layer._2+1))
      }
      else {
        layer._1.setNextLayer(outputLayer)
        layer._1.setPreviousLayer(hiddenLayer(layer._2-1))
      }
    }
    outputLayer.setPreviousLayer(hiddenLayer.last)

    var layerList : mutable.MutableList[Layer] = new mutable.MutableList[Layer]
    layerList += inputLayer
    hiddenLayer.foreach(layer => layerList += layer)
    layerList += outputLayer

    var network = Network("neuralNetwork", inputLayer, hiddenLayer, outputLayer)

    layerList.zipWithIndex.foreach{ item =>
      println(item._1.layerId)
      if(item._2 +1 != layerList.size){
        item._1.perceptron.foreach{ from =>
          println(from.perceptronId)
          println(s"next ${item._1.nextLayer.layerId}")
          item._1.nextLayer.perceptron.foreach{ to =>
            println(to.perceptronId)
            network.addSynapses(from.perceptronId, to.perceptronId, 0)
          }
        }
      }
    }

    //network

    //print synapsis


  }

}
