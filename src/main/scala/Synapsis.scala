/**
  * Created by stevanus.andrianta on 6/2/2016.
  */

case class Synapsis(fromId: String,
                    toId: String,
                    var weight: Double,
                    var lastDeltaWeight: Double = 0.00)
