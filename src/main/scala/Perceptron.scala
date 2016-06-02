/**
  * Created by stevanus.andrianta on 6/2/2016.
  */

case class Perceptron(perceptronId: String = "perceptron",
                      var activation: Double = 0.00,
                      var error: Double = 0.00,
                      isBias: Boolean = false)
