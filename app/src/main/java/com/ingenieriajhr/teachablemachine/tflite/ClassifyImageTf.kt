package com.ingenieriajhr.teachablemachine.tflite

import android.content.Context
import android.graphics.Bitmap
import android.icu.text.RelativeDateTimeFormatter.RelativeUnit
import androidx.camera.core.processing.SurfaceProcessorNode.In
import com.ingenieriajhr.teachablemachine.MainActivity
import com.ingenieriajhr.teachablemachine.MainActivity.Companion.INPUT_SIZE
import com.ingenieriajhr.teachablemachine.ml.ModelUnquant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max

class ClassifyImageTf(context:Context) {


    //get instance model classifier image
    var modelUnquant = ModelUnquant.newInstance(context)


    lateinit var returnInterpreter: ReturnInterpreter

    fun listenerInterpreter(returnInterpreter: ReturnInterpreter){
        this.returnInterpreter = returnInterpreter
    }

    fun classify(img:Bitmap){
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, INPUT_SIZE, INPUT_SIZE, 3), DataType.FLOAT32)
        val byteBuffer = ByteBuffer.allocateDirect(4* INPUT_SIZE * INPUT_SIZE *3)
        byteBuffer.order(ByteOrder.nativeOrder())

        // get 1D array of 224 * 224 pixels in image
        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)

        img!!.getPixels(intValues,0,img!!.width,0,0, img.width,img.height)

        // Reemplazar el bucle anidado con operaciones vectorizadas
        for (pixelValue in intValues) {
            byteBuffer.putFloat((pixelValue shr 16 and 0xFF) * (1f / 255f))
            byteBuffer.putFloat((pixelValue shr 8 and 0xFF) * (1f / 255f))
            byteBuffer.putFloat((pixelValue and 0xFF) * (1f / 255f))
        }
        inputFeature0.loadBuffer(byteBuffer)
        byteBuffer.clear()

        val output = modelUnquant.process(inputFeature0)
        val outputFeature = output.outputFeature0AsTensorBuffer
        val confidence = outputFeature.floatArray

        val maxPos = confidence.indices.maxByOrNull { confidence[it] }?:0

        returnInterpreter.classify(confidence, maxPos)


    }





}