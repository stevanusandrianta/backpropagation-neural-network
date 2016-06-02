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
import scala.annotation.tailrec


/**
  * Created by stevanusandrianta on 5/12/16.
  */

object NeuralNetwork {

  def initiateNetwork(dataInput: List[List[Double]], learningRate: Double, nInput: Int, nHidden: List[Int], nOutput: Int,
                      weightGeneratorOption: String, activationFunctionOption: String): Network = {

    val weightGenerator = WeightGenerator(weightGeneratorOption)
    val activationFunction = ActivationFunction(activationFunctionOption)

    Random.setSeed(Calendar.getInstance().getTimeInMillis)

    def createLayers(perceptronList: List[Int]): List[Layer] = {
      def recursive(id: Int, list: List[Layer]): List[Layer] = {
        if (id > perceptronList.size) {
          list
        } else {
          val prevLayerId = if (id == 1) None else Some("layer" + (id - 1).toString)
          val nextLayerId = if (id == perceptronList.size) None else Some("layer" + (id + 1).toString)

          val layer = Layer("layer" + id.toString, nextLayerId, prevLayerId, (0 to perceptronList(id - 1)).map { index =>
            if (index != perceptronList(id - 1)) Perceptron(s"layer_${id.toString}_perceptron_$index")
            else Perceptron(s"layer_${id.toString}_perceptron_bias", 1.00, isBias = true)
          }.toList)
          recursive(id + 1, layer :: list)
        }
      }
      recursive(1, Nil)
    }

    val totalPerceptronList = List.concat(List(nInput), nHidden, List(nOutput))
    val layers = createLayers(totalPerceptronList)

    def createSynapsis(layerList: List[Layer]): List[Synapsis] = {

      def recursive(layer: Layer, listSynapsis: List[Synapsis]): List[Synapsis] = {
        if (layer.nextLayerId.isEmpty) {
          listSynapsis
        } else {
          val nextLayer = layers.filter(_.layerId == layer.nextLayerId.get).head
          val synapsis = layer.perceptron.flatMap { fromPerceptron =>
            nextLayer.perceptron.map { toPerceptron =>
              Synapsis(fromPerceptron.perceptronId, toPerceptron.perceptronId, weightGenerator.generate)
            }
          }
          recursive(nextLayer, List.concat(synapsis, listSynapsis))
        }
      }

      recursive(layers.filter(_.prevLayerId.isEmpty).head, Nil)
    }

    val synapsises = createSynapsis(layers)
    val network = Network("neuralNetwork", layers, synapsises, activationFunction)

    network

  }

  def feedForward(network: Network, dataInput: List[Double]) = {

    //FEED FORWARD
    network.orderedLayer.foreach { layer =>
      layer.prevLayer match {
        case Some(prevLayer) =>
          layer.perceptron.foreach { perceptron =>
            if (!perceptron.isBias) {
              val newValue = network.activationFunction.activate(layer.prevLayer.get.perceptron.foldLeft(0.0) {
                (a, b) => a + b.activation * network.getConnection(b.perceptronId, perceptron.perceptronId).weight
              })
              perceptron.activation = newValue
            }
          }
        case None =>
          layer.getPerceptronsWithIndex.foreach { perceptron =>
            if (!perceptron._1.isBias) perceptron._1.activation = dataInput(perceptron._2)
          }
      }
    }

    network.outputLayer.perceptron.map(_.activation)

  }

  def initiateTraining(network: Network, dataInput: List[List[Double]], learningRate: Double, momentum: Double,
                       maxIteration: Int, minMSE: Double): Network = {

    @tailrec
    def train(network: Network, counter: Int = 0, mse: Double = Double.MaxValue): Network = {
      dataInput.foreach { data =>
        val flag = data.last

        //FEED FORWARD
        network.orderedLayer.foreach { layer =>
          layer.prevLayer match {
            case Some(prevLayer) =>
              layer.perceptron.foreach { perceptron =>
                if (!perceptron.isBias) {
                  val newValue = network.activationFunction.activate(prevLayer.perceptron.foldLeft(0.0) {
                    (a, b) => a + b.activation * network.getConnection(b.perceptronId, perceptron.perceptronId).weight
                  })
                  perceptron.activation = newValue
                }
              }
            case None =>
              layer.getPerceptronsWithIndex.foreach { perceptron =>
                if (!perceptron._1.isBias) perceptron._1.activation = data(perceptron._2)
              }
          }
        }

        //COUNT ERROR
        network.layers.foreach { layer =>

          layer.nextLayer match {
            case Some(nextLayer) =>
              layer.perceptron.foreach { perceptron =>
                val newError = perceptron.activation * (1 - perceptron.activation) * nextLayer.perceptron.foldLeft(0.0) {
                  (a, b) => a + (b.error * network.getConnection(b.perceptronId, perceptron.perceptronId).weight)
                }
                perceptron.error = newError
              }
            case None =>
              layer.perceptron.foreach { perceptron =>
                perceptron.error = perceptron.activation * (1 - perceptron.activation) * (flag - perceptron.activation)
              }
          }
        }

        //ADJUST WEIGHT
        network.orderedLayer.foreach { layer =>
          if (layer.nextLayer.isDefined) {
            layer.perceptron.foreach { from =>
              layer.nextLayer.get.perceptron.foreach { to =>
                val connection = network.getConnection(from.perceptronId, to.perceptronId)
                val deltaWeight = learningRate * to.error * from.activation
                connection.weight = connection.weight + deltaWeight + momentum * connection.lastDeltaWeight
                connection.lastDeltaWeight = deltaWeight
              }
            }
          }
        }

      }

      //COUNT MSE
      val mse = dataInput.map { data =>
        Math.pow(data.last - feedForward(network, data).head, 2)
      }.foldLeft(0.0){_ + _} / (2 * dataInput.size)

      println(s"${counter} -> MSE = $mse")

      if (maxIteration < counter || mse < minMSE) {
        network
      } else {
        train(network, counter + 1, mse)
      }
    }

    train(network, 0, 1)
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
