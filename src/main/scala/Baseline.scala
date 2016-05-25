import breeze.numerics.abs
import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.tree.RandomForest


case class PairWithCommonFriends(person1: Int, person2: Int, countArr: Seq[Int])

case class UserFriends(user: Int, friends: Array[Int])

case class AgeSex(age: Int, accCreate: Int, sex: Int, country: Int, location: Int, region: Int,
                  ageClassif: Int, dateCreateClassif: Int, yearCreateAcc: Int, yearBirth: Int)

object Baseline {

  def main(args: Array[String]) {

    val sparkConf = new SparkConf()
      .setAppName("Baseline")
    val sc = new SparkContext(sparkConf)
    val sqlc = new SQLContext(sc)

    import sqlc.implicits._

    val dataDir = if (args.length == 1) args(0) else "./"

    val graphPath = dataDir + "trainGraph"
    val reversedGraphPath = dataDir + "trainSubReversedGraph"
    val commonFriendsPath = dataDir + "cm"

    val predictionPath = dataDir + "prediction"

    val numPartitions = 200 //if test

    val numPartitionsGraph = 107

    val graph = {
      sc.textFile(graphPath)
        .map(line => {
          val lineSplit = line.split("\t")
          val user = lineSplit(0).toInt
          val friends = {
            lineSplit(1)
              .replace("{(", "")
              .replace(")}", "")
              .split("\\),\\(")
              .map(t => t.split(",")(0).toInt)
          }
          UserFriends(user, friends)
        })
    }

    graph
      .filter(userFriends => userFriends.friends.length >= 8 && userFriends.friends.length <= 1000)
      .flatMap(userFriends => userFriends.friends.map(x => (x, userFriends.user)))
      .groupByKey(numPartitions)

      .map(t => {

        var t_split = t._1.split(",") //string 12333, 0

        var newId = t_split(0) //id
        var numMask = t_split(1) //mask

        var arrMain = t._2.toArray

        val arrFr = ArrayBuffer.empty[Array[Int]]

        for (i <- 0 to (arrMain.length - 1)) {
          if (numMask.toDouble == 0.0)
            arrFr.append(Array(arrMain(i), 0))
          else
            arrFr.append(Array(arrMain(i), scala.math.log(numMask.toDouble).toInt))
        }

        UserFriendsMain(newId.toInt, arrFr.toArray)
      })

      .map(userFriends => userFriends.friends.sortBy(_ (0)))

      .filter(friends => friends.length >= 2 && friends.length <= 2000)
      .map(friends => new Tuple1(friends))
      .toDF
      .write.parquet(reversedGraphPath)

    def generatePairs(pplWithCommonFriends: Seq[Seq[Int]], numPartitions: Int, k: Int) = {

      val pairs = ArrayBuffer.empty[Array[Array[Int]]]

      var mask = pplWithCommonFriends(0)(1)

      var newArr = Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

      newArr(mask) = newArr(mask) + 1

      for (i <- 0 until pplWithCommonFriends.length) {
        if (pplWithCommonFriends(i)(0) % numPartitions == k) {
          for (j <- i + 1 until pplWithCommonFriends.length) {

            pairs.append(Array(Array(pplWithCommonFriends(i)(0), pplWithCommonFriends(j)(0)),
              newArr))
          }
        }
      }
      pairs
    }

    val commonFriendsCounts = {

      sqlc.read.parquet(reversedGraphPath)

        //TODO ?
        .map(t => t.getAs[Seq[Array[Int]]](0))
    }

    for (k <- 0 until numPartitionsGraph) {

      val commonFriendsCounts = {

        sqlc.read.parquet(reversedGraphPath)

          //TODO ?
          .map(t => generatePairs(t.getAs[Seq[Seq[Int]]](0), numPartitionsGraph, k))

          .flatMap(pair => pair.map(x => {
            var id1 = x(0)(0)
            var id2 = x(0)(1)
            (id1, id2) -> x(1)
          }))

          .reduceByKey((x, y) => {
            var ar1 = x
            var ar2 = y
            for (i <- 0 to (ar1.length - 1)) {
              ar1(i) = ar1(i) + ar2(i)
            }
            ar1
          })
          .map(t => {

            PairWithCommonFriends(t._1._1, t._1._2, t._2)
          })

          //TODO ?
          .filter(pair => {

          var ar = pair.countArr

          var allFreinds = 0

          for (i <- 0 to (ar.length - 1)) {
            allFreinds += ar(i);
          }
          allFreinds > 8
        })
      }
      commonFriendsCounts.toDF.repartition(4).write.parquet(commonFriendsPath + "/part_" + k)
    }

    val commonFriendsCounts = {
      sqlc.read.parquet("/Users/Nurislam/Documents/sna_hackatone/data/commonFriendsCountsPartitioned/part_*/")
        //TODO ?
        .map(t => t)
    }

    commonFriendsCounts foreach ((t2) => println(t2))

    commonFriendsCounts.toDF.repartition(4).write.parquet(commonFriendsPath + "/part_" + k)
  }

  step 2
  println("step 2 start")

  val commonFriendsCounts = {
    sqlc
      .read.parquet(
      commonFriendsPath + "/part_6/")
      //45
      .map(t => PairWithCommonFriends(t.getAs[Int](0), t.getAs[Int](1), t.getAs[Seq[Int]](2)))
  }

  // step 3
  println("step 3 start")

  val usersBC = sc.broadcast(graph.map(userFriends => userFriends.user).collect().toSet)

  val positives = {
    graph
      .flatMap(
        userFriends => userFriends.friends
          .filter(x => (usersBC.value.contains(x.toInt) && x > userFriends.user))
          .map(x => (userFriends.user, x) -> 1.0)
      )
  }

  // step 4
  println("step 4 start")

  val ageSex = {
    sc.textFile("/Users/Nurislam/PycharmProjects/sna_hackaton/demography_file_new")
      .map(line => {
        val lineSplit = line.trim().split("\t")

        (lineSplit(0).toInt -> AgeSex(lineSplit(1).toInt, lineSplit(2).toInt,
          lineSplit(3).toInt, lineSplit(4).toInt, lineSplit(5).toInt, lineSplit(6).toInt, lineSplit(7).toInt,
          lineSplit(8).toInt, lineSplit(9).toInt, lineSplit(10).toInt))
      })
  }

  val ageSexBC = sc.broadcast(ageSex.collectAsMap())

  // step 5
  println("step 5 start11")

  def prepareData(
                   commonFriendsCounts: RDD[PairWithCommonFriends],
                   positives: RDD[((Int, Int), Double)],
                   ageSexBC: Broadcast[scala.collection.Map[Int, AgeSex]]) = {

    commonFriendsCounts
      .map(pair => (pair.person1, pair.person2) -> (Vectors.dense(

        pair.countArr(0).toDouble +
          pair.countArr(1).toDouble +
          pair.countArr(2).toDouble +
          pair.countArr(3).toDouble +
          pair.countArr(4).toDouble +
          pair.countArr(5).toDouble +
          pair.countArr(6).toDouble +
          pair.countArr(7).toDouble +
          pair.countArr(8).toDouble +
          pair.countArr(9).toDouble +
          pair.countArr(10).toDouble +
          pair.countArr(11).toDouble +
          pair.countArr(12).toDouble +
          pair.countArr(13).toDouble +
          pair.countArr(14).toDouble +
          pair.countArr(15).toDouble +
          pair.countArr(16).toDouble +
          pair.countArr(17).toDouble +
          pair.countArr(18).toDouble +
          pair.countArr(19).toDouble,

          pair.countArr(0).toDouble,
          pair.countArr(1).toDouble,
          pair.countArr(2).toDouble,
          pair.countArr(3).toDouble,
          pair.countArr(4).toDouble,
          pair.countArr(5).toDouble,
          pair.countArr(6).toDouble,
          pair.countArr(7).toDouble,
          pair.countArr(8).toDouble,
          pair.countArr(9).toDouble,
          pair.countArr(10).toDouble,
          pair.countArr(11).toDouble,
          pair.countArr(12).toDouble,
          pair.countArr(13).toDouble,
          pair.countArr(14).toDouble,
          pair.countArr(15).toDouble,
          pair.countArr(16).toDouble,
          pair.countArr(17).toDouble,
          pair.countArr(18).toDouble,
          pair.countArr(19).toDouble,

        //разница в возрасте
        abs(ageSexBC.value.getOrElse(pair.person1, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).age
          - ageSexBC.value.getOrElse(pair.person2, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).age).toDouble,

        //одинаковые ли полы
        if (ageSexBC.value.getOrElse(pair.person1, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).sex
          == ageSexBC.value.getOrElse(pair.person2, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).sex) 1.0
        else 0.0,

        //одинаковые ли страны
        if (ageSexBC.value.getOrElse(pair.person1, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).country
          == ageSexBC.value.getOrElse(pair.person2, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).country) 1.0
        else 0.0,

        //одинаковые ли города
        if (ageSexBC.value.getOrElse(pair.person1, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).location
          == ageSexBC.value.getOrElse(pair.person2, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).location) 1.0
        else 0.0,

        //олинаковые ли регионы
        if (ageSexBC.value.getOrElse(pair.person1, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).region
          == ageSexBC.value.getOrElse(pair.person2, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).region) 1.0
        else 0.0,

        //как давно был создан аккаунт
        abs(ageSexBC.value.getOrElse(pair.person1, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).accCreate
          - ageSexBC.value.getOrElse(pair.person2, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).accCreate).toDouble,

        //соотнесение одной из возрастных категорий
        if (ageSexBC.value.getOrElse(pair.person1, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).ageClassif
          == ageSexBC.value.getOrElse(pair.person2, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).ageClassif) 1.0
        else 0.0,

        //соотнесение даты создания аккаунта одной из категорий
        if (ageSexBC.value.getOrElse(pair.person1, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).dateCreateClassif
          == ageSexBC.value.getOrElse(pair.person2, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).dateCreateClassif) 1.0
        else 0.0,

        //совпадение даты создания аккаунта
        if (ageSexBC.value.getOrElse(pair.person1, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).accCreate
          == ageSexBC.value.getOrElse(pair.person2, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).accCreate) 1.0
        else 0.0,

        //совпадение даты рождения
        if (ageSexBC.value.getOrElse(pair.person1, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).age
          == ageSexBC.value.getOrElse(pair.person2, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).age) 1.0
        else 0.0,

        //совпадение года создания аккаунта
        if (ageSexBC.value.getOrElse(pair.person1, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).yearCreateAcc
          == ageSexBC.value.getOrElse(pair.person2, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).yearCreateAcc) 1.0
        else 0.0,

        //совпадение года рождения
        if (ageSexBC.value.getOrElse(pair.person1, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).yearBirth
          == ageSexBC.value.getOrElse(pair.person2, AgeSex(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).yearBirth) 1.0
        else 0.0
      ))
      )
      .leftOuterJoin(positives)
  }

  val data = {
    prepareData(commonFriendsCounts, positives, ageSexBC)
      .map(t => LabeledPoint(t._2._2.getOrElse(0.0), t._2._1))
  }

  println("start write file")
  data.saveAsTextFile("{{data_path}}")

  val file = sc.textFile("{{data_path}}")

  val data = file.map(l => {
    val lineSplit = l.split(" ")
    LabeledPoint(lineSplit(0).toDouble, Vectors.dense(
      lineSplit(1).toDouble, //0

      lineSplit(2).toDouble, //1
      lineSplit(3).toDouble, //2
      lineSplit(4).toDouble, //3
      lineSplit(5).toDouble, //4
      lineSplit(6).toDouble, //5
      lineSplit(7).toDouble, //6
      lineSplit(8).toDouble, //7
      lineSplit(9).toDouble, //8
      lineSplit(10).toDouble, //9
      lineSplit(11).toDouble, //10
      lineSplit(12).toDouble, //11
      lineSplit(13).toDouble, //12
      lineSplit(14).toDouble, //13
      lineSplit(15).toDouble, //14
      lineSplit(16).toDouble, //15
      lineSplit(17).toDouble, //16
      lineSplit(18).toDouble, //17
      lineSplit(19).toDouble, //18
      lineSplit(20).toDouble, //19
      lineSplit(21).toDouble, //20

      lineSplit(22).toDouble, //21
      lineSplit(23).toDouble, //22 
      lineSplit(24).toDouble, //23 
      lineSplit(25).toDouble, //24 
      lineSplit(26).toDouble, //25 
      lineSplit(27).toDouble, //26
      lineSplit(28).toDouble, //27 
      lineSplit(29).toDouble, //28 
      lineSplit(30).toDouble, //29 
      lineSplit(31).toDouble, //30 
      lineSplit(32).toDouble, //31 
      lineSplit(33).toDouble //32 
    ))
  })

  // step 6
  println("step 6 start")

  val splits = data.randomSplit(Array(0.1, 0.9), seed = 11L)
  val training = splits(0).cache()
  val validation = splits(1)

  val numClasses = 2
  val categoricalFeaturesInfo = Map(22 -> 2, 23 -> 2, 24 -> 2, 25 -> 2, 27 -> 2,
    28 -> 2, 29 -> 2, 30 -> 2, 31 -> 2, 32 -> 2)
  val numTrees = 50
  // Use more in practice.
  val featureSubsetStrategy = "auto"
  // Let the algorithm choose.s
  val impurity = "variance"
  val maxDepth = 7
  val maxBins = 32

  val model = RandomForest.trainRegressor(training, categoricalFeaturesInfo,
    numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

  Train a GradientBoostedTrees model.
  The defaultParams for Regression use SquaredError by default.
  val boostingStrategy = BoostingStrategy.defaultParams("Regression")
  boostingStrategy.numIterations = 50 // Note: Use more iterations in practice.
  boostingStrategy.treeStrategy.maxDepth = 5
  // Empty categoricalFeaturesInfo indicates all features are continuous.
  boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map(2 -> 2, 3 -> 2, 4 -> 2, 5 -> 2, 7 -> 2, 8 -> 2, 9 -> 2,
    10 -> 2, 11 -> 2, 12 -> 2)

  val model = GradientBoostedTrees.train(training, boostingStrategy)

  val predictionAndLabels = {
    validation.map { case LabeledPoint(label, features) =>
      val predictionForest = model.predict(features)
      val predictionLogistic = modelLogistic.predict(features)

      val prediction = (predictionForest + predictionLogistic) / 2.0
      (prediction, label)
    }
  }

  @transient val metrics = new BinaryClassificationMetrics(predictionAndLabels)

  // 2.0 - beta factor
  val threshold = metrics.fMeasureByThreshold(2.0).sortBy(-_._2).take(1)(0)._1

  val rocLogReg = metrics.areaUnderROC()

  val precision = metrics.precisionByThreshold().sortBy(-_._2).take(1)(0)._1

  val recall = metrics.recallByThreshold().sortBy(-_._2).take(1)(0)._1

  val f1Score = metrics.fMeasureByThreshold().sortBy(-_._2).take(1)(0)._1

  val precision2 = metrics.precisionByThreshold().sortBy(-_._2).take(1)(0)._2

  val recall2 = metrics.recallByThreshold().sortBy(-_._2).take(1)(0)._2

  val f1Score2 = metrics.fMeasureByThreshold().sortBy(-_._2).take(1)(0)._2

  println("model ROC = " + rocLogReg.toString)
  println("model presision " + precision.toString + " " + precision2.toString)
  println("model recall " + recall.toString + " " + recall2.toString)
  println("model F-score " + f1Score.toString + " " + f1Score2.toString)

  // step 7
  println("step 7 start")

  val testCommonFriendsCounts = {
    sqlc
      .read.parquet(commonFriendsPath + "/part_*/")
      .map(t => PairWithCommonFriends(t.getAs[Int](0), t.getAs[Int](1), t.getAs[Seq[Int]](2)))
      .filter(pair => pair.person1 % 11 == 7 || pair.person2 % 11 == 7)
  }

  val testData = {
    prepareData(testCommonFriendsCounts, positives, ageSexBC)
      .map(t => t._1 -> LabeledPoint(t._2._2.getOrElse(0.0), t._2._1))
      .filter(t => t._2.label == 0.0)
  }

  // step 8
  println("step 8 start")

  val testPrediction = {
    testData
      .flatMap { case (id, LabeledPoint(label, features)) =>
        val prediction = model.predict(features)
        Seq(id._1 ->(id._2, prediction), id._2 ->(id._1, prediction))
      }

      .filter(t => t._1 % 11 == 7 && t._2._2 >= threshold)
      .groupByKey(numPartitions)
      .map(t => {
        val user = t._1
        val firendsWithRatings = t._2

        //changed
        val topBestFriends = firendsWithRatings.toList.sortBy(-_._2).take(100).map(x => x._1)
        (user, topBestFriends)
      })
      .sortByKey(true, 1)
      .map(t => t._1 + "\t" + t._2.mkString("\t"))
  }

  testPrediction.saveAsTextFile(predictionPath, classOf[GzipCodec])

}