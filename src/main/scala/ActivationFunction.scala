import com.fasterxml.jackson.annotation.JsonSubTypes.Type
import com.fasterxml.jackson.annotation.{JsonSubTypes, JsonTypeInfo}
import com.fasterxml.jackson.databind.annotation.{JsonDeserialize, JsonSerialize}

/**
  * Created by stevanus.andrianta on 6/2/2016.
  */

@JsonTypeInfo(
  use = JsonTypeInfo.Id.NAME,
  include = JsonTypeInfo.As.PROPERTY,
  property = "type"
)
@JsonSubTypes(Array(
  new Type(value = classOf[Sigmoid], name = "sigmoid")
))
trait ActivationFunctionTrait{
  def activate(z: Double) : Double
}

class Sigmoid extends ActivationFunctionTrait{
  override def activate(z:Double) : Double = {
    1 / (1 + Math.exp(z * (-1)))
  }
}

class Step extends ActivationFunctionTrait{
  override def activate(z:Double) : Double = {
    if(z<0) 0
    else 1
  }
}
