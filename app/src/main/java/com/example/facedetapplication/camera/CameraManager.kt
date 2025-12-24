package com.example.facedetapplication.camera

import android.content.Context
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.Executors



class CameraManager(
    private val lifecycleOwner: LifecycleOwner,
    private val context: Context,
    private val analyzer: ImageAnalysis.Analyzer
) {
    private var cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

    fun start(previewView: PreviewView) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val provider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().apply {
                setSurfaceProvider(previewView.surfaceProvider)
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            analysis.setAnalyzer(Executors.newSingleThreadExecutor(), analyzer)

            provider.unbindAll()
            provider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                analysis
            )
        }, ContextCompat.getMainExecutor(context))
    }

    fun flipCamera() {
        cameraSelector = if (cameraSelector == CameraSelector.DEFAULT_FRONT_CAMERA)
            CameraSelector.DEFAULT_BACK_CAMERA
        else
            CameraSelector.DEFAULT_FRONT_CAMERA

        // Restart camera preview with new selector
        start(previewViewGlobal)
    }

    private lateinit var previewViewGlobal: PreviewView
    fun startWithReference(previewView: PreviewView) {
        previewViewGlobal = previewView
        start(previewView)
    }
}
