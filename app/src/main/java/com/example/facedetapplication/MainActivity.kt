package com.example.facedetapplication

import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageAnalysis
import android.os.Bundle
import androidx.camera.core.ImageProxy
import com.example.facedetapplication.camera.CameraManager
import com.example.facedetapplication.db.AppDatabase
import com.example.facedetapplication.util.ImageUtils
import com.example.facedetapplication.databinding.ActivityMainBinding

abstract class MainActivity : AppCompatActivity(), ImageAnalysis.Analyzer {

    private lateinit var pipeline: RecognitionPipeline

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        pipeline = RecognitionPipeline(
            FaceDetector(this),
             LandmarkExtractor(this),
            FaceRecognition(this),
            AntiSpoof(this, "spoof_model_scale_2_7.tflite"),
            VectorStorage(AppDatabase.create(this)),
            TemporalAggregator()
        )

        CameraManager(this, this, this).start(binding.previewView)
    }

    override fun analyze(image: ImageProxy) {
        val bmp = ImageUtils.imageToBitmap(image)
        pipeline.process(bmp)
        image.close()
    }
}
