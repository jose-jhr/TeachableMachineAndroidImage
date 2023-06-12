package com.ingenieriajhr.teachablemachine.tflite

interface ReturnInterpreter {

    fun classify(confidence:FloatArray,maxConfidence:Int)

}