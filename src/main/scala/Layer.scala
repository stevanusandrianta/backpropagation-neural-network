import com.fasterxml.jackson.annotation.JsonIgnore

/**
  * Created by stevanus.andrianta on 6/2/2016.
  */
case class Layer(layerId: String, nextLayerId: Option[String] = None, prevLayerId: Option[String] = None, perceptron: List[Perceptron]) {

  @transient
  var owner: Network = _

  @JsonIgnore
  lazy val nextLayer = nextLayerId.map(owner.getLayerById(_))

  @JsonIgnore
  lazy val prevLayer = prevLayerId.map(owner.getLayerById(_))

  @JsonIgnore
  lazy val indexedPerceptron = perceptron.zipWithIndex

  @JsonIgnore
  def getPerceptronsWithIndex = indexedPerceptron

}
