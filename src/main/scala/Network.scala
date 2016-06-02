import com.fasterxml.jackson.annotation.{JsonIgnore, JsonIgnoreProperties}

/**
  * Created by stevanus.andrianta on 6/2/2016.
  */
case class Network(networkId: String,
                   layers: List[Layer],
                   synapises: List[Synapsis],
                   activationFunction: ActivationFunctionTrait) {

  layers.foreach(_.owner = this)

  private lazy val layerListMap = layers.map(i => i.layerId -> i).toMap

  def getLayerById(layerId: String) = layerListMap(layerId)

  lazy val inputLayer = layers.filter(_.prevLayer == None).head
  lazy val hiddenLayer = layers.filter(layer => layer.nextLayer != None && layer.prevLayer != None)
  lazy val outputLayer = layers.filter(_.nextLayer == None).head

  lazy val orderedLayer = layers.reverse

  lazy val synapsisMap = synapises.map(a => (a.fromId + "_" + a.toId) -> a).toMap

  def getConnection(from: String, to: String) =
    synapsisMap.get(from + "_" + to).getOrElse(synapsisMap.get(to + "_" + from).get)

}
