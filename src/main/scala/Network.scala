import java.io.{DataInput, FileWriter}
import java.util.Calendar

import jdk.internal.org.objectweb.asm.tree.MultiANewArrayInsnNode
import jdk.nashorn.internal.ir.debug.JSONWriter

import scala.collection.mutable._
import scala.io.Source
import scala.util.Random

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

  var nextLayer: Layer = null
  var prevLayer: Layer = null

  def setNextLayer(layer: Layer) = {
    this.nextLayer = layer
  }

  def setPreviousLayer(layer: Layer) = {
    this.prevLayer = layer
  }

}

case class Network(networkId: String, inputLayer: Layer, hiddenLayer: List[Layer], outputLayer: Layer) {

  var connections: MutableList[Connection] = new MutableList[Connection]

  def addConnection(connection: Connection) = {
    connections += connection
  }

  def getConnection(from: String, to: String): Connection = {
    connections.find(con => (con.fromId == from && con.toId == to)).getOrElse(
      connections.find(con => (con.fromId == to && con.toId == from)).get
    )
  }

}

case class Connection(fromId: String, toId: String, var weight: Double)

object NeuralNetwork {

  def initiateNetwork(dataInput: List[List[Double]], learningRate: Double,
                      nInput: Int, nHidden: List[Int], nOutput: Int): Network = {

    Random.setSeed(Calendar.getInstance().getTimeInMillis)

    //INITIATE LAYER AND PERCEPTRON
    var hiddenLayer = nHidden.zipWithIndex.map { item =>
      new Layer(s"hidden_layer_${item._2}", (0 until item._1).zipWithIndex.map { counter =>
        Perceptron(s"hidden_perceptron_${item._2}_${counter._2}")
      }.toList)
    }

    var inputLayer = new Layer("input", (0 until nInput).zipWithIndex.map { item =>
      Perceptron(s"input_perceptron_${item._2}")
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
            network.addConnection(new Connection(perceptronFrom.perceptronId, perceptronTo.perceptronId, lib.generateWeight))
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
      item._1.activation = dataInput(item._2)
    }

    //FEED FORWARD TO HIDDEN
    network.hiddenLayer.foreach { layer =>
      layer.perceptron.foreach { perceptron =>
        perceptron.activation = lib.sigmoid(layer.prevLayer.perceptron.foldLeft(0.0) {
          (a, b) =>  a + b.activation * network.getConnection(b.perceptronId, perceptron.perceptronId).weight
        })
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

  def initiateTraining(givenNetwork: Network, dataInput: List[List[Double]], learningRate: Double, maxIteration: Int, minMSE: Double): Network = {

    var network = givenNetwork

    (0 to maxIteration).foreach { iter =>
      dataInput.foreach { data =>
        var flag = data.last

        //INITIATE TO INPUT LAYER
        network.inputLayer.perceptron.zipWithIndex.foreach { item =>
          item._1.activation = data(item._2)
        }

        //FEED FORWARD TO HIDDEN
        network.hiddenLayer.foreach { layer =>
          layer.perceptron.foreach { perceptron =>
            perceptron.activation = lib.sigmoid(layer.prevLayer.perceptron.foldLeft(0.0) {
              (a, b) => a + b.activation * network.getConnection(b.perceptronId, perceptron.perceptronId).weight
            })
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
            var weight = network.getConnection(fromPerceptron.perceptronId, toPerceptron.perceptronId).weight
            network.getConnection(fromPerceptron.perceptronId, toPerceptron.perceptronId).weight =
              weight + learningRate * toPerceptron.error * fromPerceptron.activation
          }
        }

        //ADJUST THE WEIGHT FOR EACH HIDDEN TO OUTPUT
        network.hiddenLayer.foreach { layer =>
          layer.perceptron.foreach { fromPerceptron =>
            layer.nextLayer.perceptron.foreach { toPerceptron =>
              var weight = network.getConnection(fromPerceptron.perceptronId, toPerceptron.perceptronId).weight
              network.getConnection(fromPerceptron.perceptronId, toPerceptron.perceptronId).weight =
                weight + learningRate * toPerceptron.error * fromPerceptron.activation
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

    new FileWriter("weight.txt").flush()
    val out = new FileWriter("weight.txt", true)
    out.write("input\n")
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
    }
    out.close()

  }

  def loadNetwork = {
    val text = Source.fromFile("weight.txt").getLines().toArray
    val inputIndex = text.indexWhere(_ == "input")
    val hiddenIndex = text.indexWhere(_ == "hidden")
    val outputIndex = text.indexWhere(_ == "output")
    val synapsisIndex = text.indexWhere(_ == "synapsis")

    val inputPerceptrons : List[Perceptron] = text(inputIndex+1).split(",").toList.map(Perceptron(_))
    val hiddenPerceptrons : List[List[Perceptron]] = (hiddenIndex+1 to outputIndex-1).map{index =>
      text(index).split(",").map(Perceptron(_)).toList
    }.toList
    val outputPerceptrons : List[Perceptron] = text(outputIndex+1).split(",").toList.map(Perceptron(_))

    val synapsis : List[Connection] = (synapsisIndex+1 to text.size-1).map{index =>
      var splittedStr = text(index).split(",")
      Connection(splittedStr(0), splittedStr(1), splittedStr(2).toDouble)
    }.toList

    var hiddenLayer = hiddenPerceptrons.zipWithIndex.map{ perceptronList =>
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
    synapsis.foreach{ con =>
      network.addConnection(con)
    }

    network

  }

}
