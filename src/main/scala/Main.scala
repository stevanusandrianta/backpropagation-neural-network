/**
  * Created by stevanusandrianta on 4/21/16.
  */

import BackPropagation._

object Main {

  val dataInput : List[List[Double]] = List(
    List(1,2,0),
    List(3,1,0),
    List(2,2,0),
    List(3,5,1),
    List(5,4,1),
    List(4,5,1),
    List(3,4,1),
    List(2,3,0),
    List(1,0,0),
    List(5,3,1)
  )

  def main(args: Array[String]): Unit = {

    println("Start Of Neural Network Algorithm")
    BackPropagation.bpnn(dataInput, 0.05, 2, 5, 1)

  }

}
