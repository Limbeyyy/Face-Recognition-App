package com.example.facedetapplication.model

import android.graphics.Rect
data class FaceBox(
    val bbox: Rect,
    val confidence: Float
)
