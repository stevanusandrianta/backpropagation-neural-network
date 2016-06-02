/**
  * Created by stevanus.andrianta on 6/2/2016.
  */

trait ActivationFunctionTrait{
  def activate(z: Double) : Double
}

object Sigmoid extends ActivationFunctionTrait{
  override def activate(z:Double) : Double = {
    1 / (1 + Math.exp(z * (-1)))
  }
}

object Step extends ActivationFunctionTrait{
  override def activate(z:Double) : Double = {
    if(z<0) 0
    else 1
  }
}
