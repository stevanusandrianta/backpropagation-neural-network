import scala.util.Random

/**
  * Created by stevanus.andrianta on 6/2/2016.
  */

trait WeightGeneratorTrait {
  def generate : Double
}

object WeightGenerator1 extends WeightGeneratorTrait{
  override def generate: Double = {
    val random = Random.nextDouble() / 10
    if (random > 0.05)
      random - 0.1
    else random
  }
}

object WeightGenerator2 extends WeightGeneratorTrait{
  override def generate: Double = {
    val random = Random.nextDouble() * 6 / 10
    if (random > 0.3)
      random - 0.6
    else random
  }
}




