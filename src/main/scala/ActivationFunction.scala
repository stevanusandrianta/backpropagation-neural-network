/**
  * Created by stevanus.andrianta on 6/2/2016.
  */

case class ActivationFunction(option: String = "sigmoid"){

  def sigmoid(z: Double): Double = {
    1 / (1 + Math.exp(z * (-1)))
  }

  def step(z: Double): Double = {
    if(z<0) 0
    else 1
  }

  def activate(z: Double): Double = {
    option match {
      case "sigmoid" => sigmoid(z)
      case "step" => step(z)
      case _ => sigmoid(z)
    }
  }

}
