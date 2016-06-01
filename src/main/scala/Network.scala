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

  @JsonIgnore
  var nextLayer: Layer = null

  @JsonIgnore
  var prevLayer: Layer = null

  def setNextLayer(layer: Layer) = {
    this.nextLayer = layer
  }

  def setPreviousLayer(layer: Layer) = {
    this.prevLayer = layer
  }

}

case class Network(networkId: String, inputLayer: Layer, hiddenLayer: List[Layer], outputLayer: Layer) {

  var layerList = new mutable.HashMap[String, Layer]()
  def getLayerById(layerId: String) = layerList.get(layerId)

  var connections: MutableList[Synapsis] = new MutableList[Synapsis]

  def addConnection(connection: Synapsis) = {
    connections += connection
  }

  def getConnection(from: String, to: String): Synapsis = {
    connections.find(con => (con.fromId == from && con.toId == to)).getOrElse(
      connections.find(con => (con.fromId == to && con.toId == from)).get
    )
  }

}

case class Synapsis(fromId: String, toId: String, var weight: Double, var lastDeltaWeight: Double = 0.00)

object NeuralNetwork {

  def initiateNetwork(dataInput: List[List[Double]], learningRate: Double,
                      nInput: Int, nHidden: List[Int], nOutput: Int): Network = {

    Random.setSeed(Calendar.getInstance().getTimeInMillis)

    //INITIATE LAYER AND PERCEPTRON
    var hiddenLayer = nHidden.zipWithIndex.map { item =>
      new Layer(s"hidden_layer_${item._2}", (0 to item._1).zipWithIndex.map { counter =>
        if (counter._2 != item._1) {
          Perceptron(s"hidden_perceptron_${item._2}_${counter._2}")
        } else {
          Perceptron(s"hidden_bias_${item._2}", 1.00)
        }
      }.toList)
    }

    var inputLayer = new Layer("input", (0 to nInput).zipWithIndex.map { item =>
      if (item._2 != nInput) {
        Perceptron(s"input_perceptron_${item._2}")
      } else {
        Perceptron("input_bias", 1.00)
      }
    }.toList)

    var outputLayer = new Layer("output", (0 until nOutput).zipWithIndex.map { item =>
      Perceptron(s"output_perceptron_${item._2}")
    }.toList)

    //SETUP LAYERING NETWORK
    inputLayer.nextLayer = hiddenLayer.head
    hiddenLayer.zipWithIndex.foreach { layer =>
      if (hiddenLayer.size == 1) {
        layer._1.setNextLayer(outputLayer)
        layer._1.setPreviousLayer(inputLayer)
      } else if (layer._2 == 0) {
        layer._1.setNextLayer(hiddenLayer(layer._2 + 1))
        layer._1.setPreviousLayer(inputLayer)
      }
      else {
        layer._1.setNextLayer(outputLayer)
        layer._1.setPreviousLayer(hiddenLayer(layer._2 - 1))
      }
    }
    outputLayer.setPreviousLayer(hiddenLayer.last)

    var network = Network("neuralNetwork", inputLayer, hiddenLayer, outputLayer)

    //COMBINING ALL LAYER
    var layerList: MutableList[Layer] = new MutableList[Layer]
    layerList += inputLayer
    hiddenLayer.foreach(layer => layerList += layer)
    layerList += outputLayer

    //ADD CONNECTION TO NETWORK
    layerList.zipWithIndex.foreach { item =>
      if (item._2 + 1 != layerList.size) {
        item._1.perceptron.foreach { perceptronFrom =>
          item._1.nextLayer.perceptron.foreach { perceptronTo =>
            network.addConnection(new Synapsis(perceptronFrom.perceptronId, perceptronTo.perceptronId, lib.generateWeight))
          }
        }
      }
    }

    //RETURN THE NETWORK
    network
  }

  def feedForward(network: Network, dataInput: List[Double]) = {

    var flag = dataInput.last

    //INITIATE TO INPUT LAYER
    network.inputLayer.perceptron.zipWithIndex.foreach { item =>
      if (!item._1.perceptronId.contains("bias"))
        item._1.activation = dataInput(item._2)
    }

    //FEED FORWARD TO HIDDEN
    network.hiddenLayer.foreach { layer =>
      layer.perceptron.foreach { perceptron =>
        if (!perceptron.perceptronId.contains("bias")) {
          perceptron.activation = lib.sigmoid(layer.prevLayer.perceptron.foldLeft(0.0) {
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

    network.outputLayer.perceptron.map(_.activation)

  }

  def initiateTraining(givenNetwork: Network, dataInput: List[List[Double]], learningRate: Double, momentum: Double, maxIteration: Int, minMSE: Double): Network = {

    var network = givenNetwork

    (0 to maxIteration).foreach { iter =>
      dataInput.foreach { data =>
        var flag = data.last

        //INITIATE TO INPUT LAYER
        network.inputLayer.perceptron.zipWithIndex.foreach { item =>
          if (!item._1.perceptronId.contains("bias"))
            item._1.activation = data(item._2)
        }

        //FEED FORWARD TO HIDDEN
        network.hiddenLayer.foreach { layer =>
          layer.perceptron.foreach { perceptron =>
            if (!perceptron.perceptronId.contains("bias")) {
              perceptron.activation = lib.sigmoid(layer.prevLayer.perceptron.foldLeft(0.0) {
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
            var sumWeightMultipllyError =
              perceptron.error = perceptron.activation * (1 - perceptron.activation) * layer.nextLayer.perceptron.foldLeft(0.0) {
                (a, b) => a + (b.error * network.getConnection(b.perceptronId, perceptron.perceptronId).weight)
              }
          }
        }

        //ADJUST THE WEIGHT INPUT TO HIDDEN
        network.inputLayer.perceptron.foreach { fromPerceptron =>
          network.inputLayer.nextLayer.perceptron.foreach { toPerceptron =>
            var connection = network.getConnection(fromPerceptron.perceptronId, toPerceptron.perceptronId)
            var deltaWeight = learningRate * toPerceptron.error * fromPerceptron.activation
            connection.weight = connection.weight + deltaWeight + momentum * connection.lastDeltaWeight
            connection.lastDeltaWeight = deltaWeight
          }
        }

        //ADJUST THE WEIGHT FOR EACH HIDDEN TO OUTPUT
        network.hiddenLayer.foreach { layer =>
          layer.perceptron.foreach { fromPerceptron =>
            layer.nextLayer.perceptron.foreach { toPerceptron =>
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
    /*out.write("input\n")
    network.inputLayer.perceptron.foreach { p =>
      if (p != network.inputLayer.perceptron.last) out.write(s"${p.perceptronId},")
      else out.write(s"${p.perceptronId}\n")
    }
    out.write("hidden\n")
    network.hiddenLayer.foreach { layer =>
      layer.perceptron.foreach { p =>
        if (p != layer.perceptron.last) out.write(s"${p.perceptronId},")
        else out.write(s"${p.perceptronId}\n")
      }
    }
    out.write("output\n")
    network.outputLayer.perceptron.foreach { p =>
      if (p != network.outputLayer.perceptron.last) out.write(s"${p.perceptronId},")
      else out.write(s"${p.perceptronId}\n")
    }
    out.write("synapsis\n")
    network.connections.foreach { con =>
      out.write(s"${con.fromId},${con.toId},${con.weight}\n")
    }*/

    val mapper = new ObjectMapper()
    mapper.registerModule(DefaultScalaModule)

    val stringWriter = new StringWriter
    mapper.writeValue(stringWriter, network)
    val json = stringWriter.toString

    out.write(json)
    out.close()

  }

  def loadNetwork = {
    /*val commandList = List("input", "hidden", "output", "synapsis")
    //val text = Source.fromFile("weight.txt").getLines().toList

    var inputStr = new MutableList[String]
    var hiddenStr = new MutableList[String]
    var outputStr = new MutableList[String]
    var synapsisStr = new MutableList[String]
    var temp = ""
    val text = Source.fromFile("weight.txt").getLines().foreach { str =>
      if (commandList.contains(str)) {
        temp = str
      }
      else {
        temp match {
          case "input" =>
            inputStr += str
          case "hidden" =>
            hiddenStr += str
          case "output" =>
            outputStr += str
          case "synapsis" =>
            synapsisStr += str
        }
      }
    }
    
    val inputPerceptrons: List[Perceptron] = inputStr.head.trim().split(",").map(Perceptron(_)).toList
    val hiddenPerceptrons: List[List[Perceptron]] = hiddenStr.map { str =>
      str.split(",").map(Perceptron(_)).toList
    }.toList
    val outputPerceptrons: List[Perceptron] = outputStr.head.trim().split(",").map(Perceptron(_)).toList
    val synapsis: List[Connection] = synapsisStr.map { str =>
      val splittedStr = str.split(",")
      Connection(splittedStr(0), splittedStr(1), splittedStr(2).toDouble)
    }.toList

    var hiddenLayer = hiddenPerceptrons.zipWithIndex.map { perceptronList =>
      new Layer(s"hidden_layer_${perceptronList._2}", perceptronList._1)
    }
    var inputLayer = new Layer("input", inputPerceptrons)
    var outputLayer = new Layer("output", outputPerceptrons)

    inputLayer.nextLayer = hiddenLayer.head
    hiddenLayer.zipWithIndex.foreach { layer =>
      if (hiddenLayer.size == 1) {
        layer._1.setNextLayer(outputLayer)
        layer._1.setPreviousLayer(inputLayer)
      } else if (layer._2 == 0) {
        layer._1.setNextLayer(hiddenLayer(layer._2 + 1))
        layer._1.setPreviousLayer(inputLayer)
      }
      else {
        layer._1.setNextLayer(outputLayer)
        layer._1.setPreviousLayer(hiddenLayer(layer._2 - 1))
      }
    }
    outputLayer.setPreviousLayer(hiddenLayer.last)

    var network = Network("neuralNetwork", inputLayer, hiddenLayer, outputLayer)
    synapsis.foreach { con =>
      network.addConnection(con)
    }*/

    val mapper = new ObjectMapper()
    mapper.registerModule(DefaultScalaModule)

    var network = mapper.readValue(new FileReader("weight.json"),classOf[Network])

    network.inputLayer.nextLayer = network.hiddenLayer.head
    network.hiddenLayer.zipWithIndex.foreach { layer =>
      if (network.hiddenLayer.size == 1) {
        layer._1.setNextLayer(network.outputLayer)
        layer._1.setPreviousLayer(network.inputLayer)
      } else if (layer._2 == 0) {
        layer._1.setNextLayer(network.hiddenLayer(layer._2 + 1))
        layer._1.setPreviousLayer(network.inputLayer)
      }
      else {
        layer._1.setNextLayer(network.outputLayer)
        layer._1.setPreviousLayer(network.hiddenLayer(layer._2 - 1))
      }
    }
    network.outputLayer.setPreviousLayer(network.hiddenLayer.last)

    network

  }

}
