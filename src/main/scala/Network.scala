import java.io.{DataInput, FileReader, FileWriter, StringWriter}
import java.util.Calendar

import com.fasterxml.jackson.annotation.JsonIgnore
import jdk.internal.org.objectweb.asm.tree.MultiANewArrayInsnNode
import jdk.nashorn.internal.ir.debug.JSONWriter

import scala.collection.mutable
import scala.collection.mutable._
import scala.io.Source
import scala.util.Random
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule

/**
  * Created by stevanusandrianta on 5/12/16.
  */

//FUNCTION LIST
object lib {

  Random.setSeed(Calendar.getInstance().getTimeInMillis)

  def generateWeight: Double = {
    val random = Random.nextDouble() / 10
    if (random > 0.05)
      random - 0.1
    else random
  }

  def sigmoid(z: Double): Double = {
    1 / (1 + Math.exp(z * (-1)))
  }
}

case class Perceptron(perceptronId: String = "perceptron", var activation: Double = 0.00, var error: Double = 0.00)

case class Layer(layerId: String, perceptron: List[Perceptron]) {

  var nextLayerId = ""
  var prevLayerId = ""

  def setNextLayer(layer: String) = {
    this.nextLayerId = layer
  }

  def setPreviousLayer(layer: String) = {
    this.prevLayerId = layer
  }

  lazy val indexedPerceptron = perceptron.zipWithIndex
  def getPerceptronsWithIndex = indexedPerceptron

}

case class Network(networkId: String, inputLayer: Layer, hiddenLayer: List[Layer], outputLayer: Layer) {

  //private lazy val layerList = inputLayer :: outputLayer :: hiddenLayer
  private lazy val orderedLayer = List.concat(List(inputLayer), hiddenLayer, List(outputLayer))
  private lazy val layerListMap = orderedLayer.map(i => i.layerId -> i).toMap

  def getLayerById(layerId: String) = layerListMap(layerId)

  private lazy val synapsisMap = connections.map(a => (a.fromId + "_" + a.toId) -> a).toMap

  def getConnection(from: String, to: String) =
    synapsisMap.get(from + "_" + to).getOrElse(synapsisMap.get(to + "_" + from).get)

  var connections: MutableList[Synapsis] = new MutableList[Synapsis]

  def addConnection(connection: Synapsis) = {
    connections += connection
  }

}

case class Synapsis(fromId: String, toId: String, var weight: Double, var lastDeltaWeight: Double = 0.00)

object NeuralNetwork {

  def initiateNetwork(dataInput: List[List[Double]], learningRate: Double,
                      nInput: Int, nHidden: List[Int], nOutput: Int): Network = {

    Random.setSeed(Calendar.getInstance().getTimeInMillis)

    //INITIATE LAYER AND PERCEPTRON
    var hiddenLayer = nHidden.zipWithIndex.map { item =>
      new Layer(s"hidden_layer_${item._2}", (0 to item._1).map { counter =>
        if (counter != item._1) {
          Perceptron(s"hidden_perceptron_${item._2}_${counter}")
        } else {
          Perceptron(s"hidden_bias_${item._2}", 1.00)
        }
      }.toList)
    }

    var inputLayer = new Layer("input_layer", (0 to nInput).map { counter =>
      if (counter != nInput) {
        Perceptron(s"input_perceptron_${counter}")
      } else {
        Perceptron("input_bias", 1.00)
      }
    }.toList)

    var outputLayer = new Layer("output_layer", (0 until nOutput).map { counter =>
      Perceptron(s"output_perceptron_${counter}")
    }.toList)

    //SETUP LAYERING NETWORK
    inputLayer.setNextLayer(hiddenLayer.head.layerId)
    hiddenLayer.zipWithIndex.foreach { layer =>
      if (hiddenLayer.size == 1) {
        layer._1.setNextLayer(outputLayer.layerId)
        layer._1.setPreviousLayer(inputLayer.layerId)
      } else if (layer._2 == 0) {
        layer._1.setNextLayer(hiddenLayer(layer._2 + 1).layerId)
        layer._1.setPreviousLayer(inputLayer.layerId)
      }
      else {
        layer._1.setNextLayer(outputLayer.layerId)
        layer._1.setPreviousLayer(hiddenLayer(layer._2 - 1).layerId)
      }
    }
    outputLayer.setPreviousLayer(hiddenLayer.last.layerId)

    var network = Network("neuralNetwork", inputLayer, hiddenLayer, outputLayer)

    //COMBINING ALL LAYER
    val layerList = List.concat(List(inputLayer), hiddenLayer, List(outputLayer))

    //ADD CONNECTION TO NETWORK
    layerList.filterNot(_ == layerList.last).foreach { layer =>
      layer.perceptron.foreach { perceptronFrom =>
        val nextLayer = network.getLayerById(layer.nextLayerId)
        nextLayer.perceptron.foreach { perceptronTo =>
          network.addConnection(new Synapsis(perceptronFrom.perceptronId, perceptronTo.perceptronId, lib.generateWeight))
        }
      }
    }

    //RETURN THE NETWORK
    network
  }

  def feedForward(network: Network, dataInput: List[Double]) = {

    var flag = dataInput.last

    //INITIATE TO INPUT LAYER
    network.inputLayer.getPerceptronsWithIndex.foreach { item =>
      if (!item._1.perceptronId.contains("bias"))
        item._1.activation = dataInput(item._2)
    }

    //FEED FORWARD TO HIDDEN
    network.hiddenLayer.foreach { layer =>
      layer.perceptron.foreach { perceptron =>
        if (!perceptron.perceptronId.contains("bias")) {
          var prevLayer = network.getLayerById(layer.prevLayerId)
          var newValue = lib.sigmoid(prevLayer.perceptron.foldLeft(0.0) {
            (a, b) => a + b.activation * network.getConnection(b.perceptronId, perceptron.perceptronId).weight
          })
          perceptron.activation = newValue
        }
      }
    }

    //FEED FORWARD TO OUTPUT
    network.outputLayer.perceptron.foreach { perceptron =>
      perceptron.activation = lib.sigmoid(network.hiddenLayer.last.perceptron.foldLeft(0.0) {
        (a, b) => a + b.activation * network.getConnection(b.perceptronId, perceptron.perceptronId).weight
      })
    }

    network.outputLayer.perceptron.map(_.activation)

  }

  def initiateTraining(givenNetwork: Network, dataInput: List[List[Double]], learningRate: Double, momentum: Double, maxIteration: Int, minMSE: Double): Network = {

    var network = givenNetwork

    (0 to maxIteration).foreach { iter =>
      dataInput.foreach { data =>
        var flag = data.last

        //INITIATE TO INPUT LAYER
        network.inputLayer.getPerceptronsWithIndex.foreach { item =>
          if (!item._1.perceptronId.contains("bias"))
            item._1.activation = data(item._2)
        }

        //FEED FORWARD TO HIDDEN
        network.hiddenLayer.foreach { layer =>
          layer.perceptron.foreach { perceptron =>
            if (!perceptron.perceptronId.contains("bias")) {
              var prevLayer = network.getLayerById(layer.prevLayerId)
              perceptron.activation = lib.sigmoid(prevLayer.perceptron.foldLeft(0.0) {
                (a, b) => a + b.activation * network.getConnection(b.perceptronId, perceptron.perceptronId).weight
              })
            }
          }
        }

        //FEED FORWARD TO OUTPUT
        network.outputLayer.perceptron.foreach { perceptron =>
          perceptron.activation = lib.sigmoid(network.hiddenLayer.last.perceptron.foldLeft(0.0) {
            (a, b) => a + b.activation * network.getConnection(b.perceptronId, perceptron.perceptronId).weight
          })
        }

        //COUNT ERROR IN OUTPUT
        network.outputLayer.perceptron.foreach { perceptron =>
          perceptron.error = perceptron.activation * (1 - perceptron.activation) * (flag - perceptron.activation)
        }

        //COUNT ERROR IN HIDDEN
        network.hiddenLayer.reverse.foreach { layer =>
          layer.perceptron.foreach { perceptron =>
            var nextLayer = network.getLayerById(layer.nextLayerId)
            var sumWeightMultipllyError =
              perceptron.error = perceptron.activation * (1 - perceptron.activation) * nextLayer.perceptron.foldLeft(0.0) {
                (a, b) => a + (b.error * network.getConnection(b.perceptronId, perceptron.perceptronId).weight)
              }
          }
        }

        //ADJUST THE WEIGHT INPUT TO HIDDEN
        network.inputLayer.perceptron.foreach { fromPerceptron =>
          var nextLayer = network.getLayerById(network.inputLayer.nextLayerId)
          nextLayer.perceptron.foreach { toPerceptron =>
            var connection = network.getConnection(fromPerceptron.perceptronId, toPerceptron.perceptronId)
            var deltaWeight = learningRate * toPerceptron.error * fromPerceptron.activation
            connection.weight = connection.weight + deltaWeight + momentum * connection.lastDeltaWeight
            connection.lastDeltaWeight = deltaWeight
          }
        }

        //ADJUST THE WEIGHT FOR EACH HIDDEN TO OUTPUT
        network.hiddenLayer.foreach { layer =>
          var nextLayer = network.getLayerById(layer.nextLayerId)
          layer.perceptron.foreach { fromPerceptron =>
            nextLayer.perceptron.foreach { toPerceptron =>
              var connection = network.getConnection(fromPerceptron.perceptronId, toPerceptron.perceptronId)
              var deltaWeight = learningRate * toPerceptron.error * fromPerceptron.activation
              connection.weight = connection.weight + deltaWeight + momentum * connection.lastDeltaWeight
              connection.lastDeltaWeight = deltaWeight
            }
          }
        }
      }

      //COUNT MSE
      var mse = dataInput.map { data =>
        Math.pow(data.last - feedForward(network, data).head, 2)
      }.foldLeft(0.0) {
        _ + _
      } / (2 * dataInput.size)

      println(s"${iter} -> MSE = $mse")

    }

    network
  }

  def saveNetwork(network: Network) = {

    new FileWriter("weight.json").flush()
    val out = new FileWriter("weight.json", true)

    val mapper = new ObjectMapper()
    mapper.registerModule(DefaultScalaModule)

    val stringWriter = new StringWriter
    mapper.writeValue(stringWriter, network)
    val json = stringWriter.toString

    out.write(json)
    out.close()

  }

  def loadNetwork = {

    val mapper = new ObjectMapper()
    mapper.registerModule(DefaultScalaModule)

    mapper.readValue(new FileReader("weight.json"), classOf[Network])

  }

}
