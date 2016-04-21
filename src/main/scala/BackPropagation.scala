import java.util
import java.util.Calendar

import scala.collection.mutable.MutableList
import scala.util.Random

object BackPropagation {

  def generateWeight : Double = {
    val random = Random.nextDouble() / 10
    if(random > 0.05)
      random - 0.1
    else random
  }

  def sigmoid(z : Double) : Double ={
    1 / (1 + Math.exp(z * (-1)))
  }

  def bpnn(dataInput: List[List[Double]], learningRate : Double, nInput : Int, nHidden : Int, nOutput : Int): Unit = {

    Random.setSeed(Calendar.getInstance().getTimeInMillis)
    var outputAndExpectedOutput = MutableList[MutableList[Double]]()

    //INITIALIZE VARIABLE
    var w1 = Array.ofDim[Double](nHidden, nInput)
    for (i <- 0 until nHidden) {
      for (j <- 0 until nInput) {
        w1(i)(j) = generateWeight
      }
    }

    var w2 = Array.ofDim[Double](nHidden)
    for (i <- 0 until nHidden) w2(i) += generateWeight

    //INITIALIZE ACITIVATION 1 AND 2
    var act1 = MutableList[Double]()
    for (i <- 1 to nHidden) act1 += 0
    var act2: Double = 0

    //INITIALIZE ERROR 1 AND 2
    var error1 = MutableList[Double]()
    for (i <- 1 to nHidden) error1 += 0
    var error2: Double = 0

    //FOREACH THE DATA
    dataInput.foreach { data =>
      val x = List(data(0), data(1))
      val y = data(2)

      //CALCULATE ACTIVATION 1
      var temp: Double = 0
      for (i <- 0 to nHidden - 1) {
        temp = 0
        for (j <- 0 until w1(i).length) {
          temp += w1(i)(j) * x(j)
        }
        act1(i) = sigmoid(temp)
      }

      //CALCULATE ACTIVATION 2
      temp = 0
      for (i <- 0 until nHidden) {
        temp += w2(i) * act1(i)
      }
      act2 = sigmoid(temp)

      //CALCULATE ERROR 2
      error2 = act2 * (1 - act2) * (y - act2)

      //CALCULATE ERROR 1
      for (i <- 0 until nHidden) error1(i) = act1(i) * (1 - act1(i)) * (w2(i) * error2)

      //ADJUST WEIGHT 2 (HIDDEN -> OUTPUT)
      for (i <- 0 until w2.length) w2(i) = w2(i) + (learningRate * error2 * act1(i))

      //ADJUST WEITGHT 1 (INPUT -> HIDDEN)
      for (i <- 0 until w1.length) {
        for (j <- 0 until w1(i).length) {
          w1(i)(j) = w1(i)(j) + (learningRate * error1(i) * x(j))
        }
      }

      outputAndExpectedOutput += MutableList[Double](y,act2)
      println(outputAndExpectedOutput.map{
        dat => Math.pow(dat(0) - dat(1), 2)
      }.foldLeft(1D){_ + _} / outputAndExpectedOutput.length)
    }

  }
}
