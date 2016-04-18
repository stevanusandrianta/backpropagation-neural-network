import java.util

import scala.collection.mutable.MutableList

object Hello {

  val dataInput = List(
    List(1,2,0),
    List(3,1,0),
    List(2,2,0),
    List(3,5,1),
    List(5,4,1)
  )

  def generateWeight : Double = {
    val random = (Math.random()/10)
    if(random > 0.05)
      random - 0.1
    else random
  }

  def sigmoid(z : Double) : Double ={
    return(1/(1 + Math.exp(z * (-1))))
  }

  def bpnn(learningRate : Double, nInput : Int, nHidden : Int, nOutput : Int): Unit ={

    //INITIALIZE VARIABLE
    var w1 = MutableList[MutableList[Double]]()
    var tempArray = MutableList[Double]()
    for(i <- 1 to nHidden){
      tempArray.clear()
      for(j <- 1 to nInput){
        tempArray += generateWeight
      }
      w1 += tempArray
    }

    var w2 = MutableList[Double]()
    for(i <- 1 to nHidden) w2 += generateWeight

    println(w1)
    println(w2)


    //INITIALIZE ACITIVATION 1 AND 2
    var act1 = MutableList[Double]()
    for(i <- 1 to nHidden) act1 += 0
    var act2 : Double = 0

    //INITIALIZE ERROR 1 AND 2
    var error1 = MutableList[Double]()
    for(i <- 1 to nHidden) error1 += 0
    var error2 : Double = 0

    //FOREACH THE DATA
    dataInput.foreach{ data =>
      val x = List(data(0), data(1))
      val y = data(2)

      //CALCULATE ACTIVATION 1
      var temp : Double = 0
      for(i <- 0 to nHidden-1){
        temp = 0
        for(j <- 0 to w1(i).length - 1){
          temp += w1(i)(j) * x(j)
        }
        act1(i) = sigmoid(temp)
        println(act1(i))
      }

      //CALCULATE ACTIVATION 2
      temp = 0
      for(i <- 0 to nHidden-1){
        temp += w2(i) * act1(i)
      }
      act2 = sigmoid(temp)

      //CALCULATE ERROR 2
      error2 = act2 * (1 - act2) * (y - act2)

      //CALCULATE ERROR 1
      for(i <- 0 to nHidden - 1) error1(i) = act1(i) * (1 - act1(i)) * (w2(i) * error2)

      //ADJUST WEIGHT 2 (HIDDEN -> OUTPUT)
      for(i <- 0 to w2.length - 1) w2(i) = w2(i) + (learningRate * error2 * act1(i))

      //ADJUST WEITGHT 1 (INPUT -> HIDDEN)
      for(i <- 0 to w1.length - 1){
        for(j <- 0 to w1(i).length - 1){
          w1(i)(j) = w1(i)(j) + (learningRate * error1(i) * x(j))
        }
      }

      return 0

    }


  }

  def main(args: Array[String]): Unit = {

    println("Start Of Neural Network Algorithm")
    bpnn(0.05,2,3,1)

  }
}
