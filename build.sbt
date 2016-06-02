name := """hello-scala"""

version := "1.0"

scalaVersion := "2.11.7"

libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.4" % "test"

libraryDependencies += "com.fasterxml.jackson.module" % "jackson-module-scala_2.11" % "2.7.3"

//http://mvnrepository.com/artifact/com.fasterxml.jackson.module/jackson-module-scala_2.11



fork in run := true