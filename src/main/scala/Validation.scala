/**
  * Created by stevanus.andrianta on 5/11/2016.
  */
object Validation {

  def splitValidation(data: List[List[Double]], training: Double, learningRate : Double, nInput : Int, nHidden : Int, nOutput : Int, maxIter: Int) = {
    val splittedData = data.splitAt((training * data.size).round.toInt)
    val weight = BackPropagation.trainBPNN(splittedData._1, learningRate, nInput, nHidden, nOutput, maxIter)

    val comparedData = splittedData._2.zipWithIndex.map{ item =>
      val expectedValue = item._1.last
      val classificationValue = BackPropagation.classifyBPNN(weight._1, weight._2, item._1)
      println(s"should be ${expectedValue} prediction be ${classificationValue}")
      if(expectedValue == classificationValue) true else false
    }

    val trueClassification = comparedData.count(_ == true)
    val falseClassification = comparedData.count(_ == false)
    println(s"total data : ${splittedData._2.size}, correct : $trueClassification, false : $falseClassification")

  }

  def crossValidation(data: List[List[Double]], group: Int, learningRate : Double, nInput : Int, nHidden : Int, nOutput : Int, maxIter: Int) = {
    val splitter : Double = data.size / group.toDouble
    (1 to group).map{ i =>
      val start = ((i-1)*splitter).round.toInt
      val end = (i*splitter).round.toInt
      val sequences = (start until end).toList
      println(sequences)

      val trainingData = data.zipWithIndex.filterNot(item => sequences.contains(item._2)).map(_._1)
      val testData = data.zipWithIndex.filter(item => sequences.contains(item._2)).map(_._1)

      println(trainingData)

      val weight = BackPropagation.trainBPNN(trainingData, learningRate, nInput, nHidden, nOutput, maxIter)

      val comparedData = testData.zipWithIndex.map{ item =>
        val expectedValue = item._1.last
        val classificationValue = BackPropagation.classifyBPNN(weight._1, weight._2, item._1).round.toDouble
        if(expectedValue == classificationValue) true else false
      }

      val trueClassification = comparedData.filter(_ == true).size
      val falseClassification = comparedData.filter(_ == false).size
      println(s"iteration number : $i, total data : ${testData.size}, correct : $trueClassification, false : $falseClassification")

    }
  }


}
