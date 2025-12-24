package com.example.facedetapplication

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.example.facedetapplication.camera.CameraManager
import com.example.facedetapplication.databinding.ActivityMainBinding
import com.example.facedetapplication.db.AppDatabase
import com.example.facedetapplication.util.ImageUtils
import android.content.Intent
import kotlin.jvm.java

class MainActivity : AppCompatActivity(), ImageAnalysis.Analyzer {

    private lateinit var pipeline: RecognitionPipeline
    private lateinit var binding: ActivityMainBinding



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        pipeline = RecognitionPipeline(
            FaceDetector(this),
            LandmarkExtractor(this),
            FaceRecognition(this),
            AntiSpoof(this, "spoof_model_scale_2_7.tflite"),
            VectorStorage(AppDatabase.create(this)),
            TemporalAggregator()
        )

        CameraManager(
            context = this,
            lifecycleOwner = this,
            analyzer = this
        ).start(binding.previewView)

        // Add Face button
        binding.btnAddFace.setOnClickListener {
            val intent = Intent(this, FaceActivity::class.java)
            startActivity(intent)
        }

        // Face List button
        binding.btnFaceList.setOnClickListener {
            val intent = Intent(this, FaceListActivity::class.java)
            startActivity(intent)
        }

    }

    override fun analyze(image: ImageProxy) {
        val bmp = ImageUtils.imageToBitmap(image)
        pipeline.process(bmp)
        image.close()
    }
}
