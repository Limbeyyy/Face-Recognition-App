package com.example.facedetapplication

import android.graphics.Bitmap
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.facedetapplication.db.AppDatabase
import com.example.facedetapplication.util.BlurEstimator
import kotlin.collections.firstOrNull
import com.example.facedetapplication.model.FaceBox



class EnrollActivity : AppCompatActivity() {

    private lateinit var storage: VectorStorage

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        storage = VectorStorage(AppDatabase.create(this))
    }

    /**
     * Enrolls a single face image under the given person name
     */
    fun enroll(person: String, bitmap: Bitmap) {
        val detector = FaceDetector(this)
        val embedder = FaceRecognition(this)
        val landmark = LandmarkExtractor(this)

        // 1. Detect faces
        val face: FaceBox = detector.detect(bitmap).firstOrNull() ?: return

        // 2. Extract landmarks
        val lm = landmark.extract(bitmap, face.bbox)
        val landmarkWeight = landmark.quality(lm)

        // 3. Blur weight
        val blurWeight = BlurEstimator.score(bitmap, face.bbox)

        // 4. Combined weight
        val weight = face.confidence * landmarkWeight * blurWeight

        // 5. Embed face
        val embedding = embedder.embed(bitmap, face.bbox)

        // 6. Store in vector storage
        storage.add(person, embedding, weight)
    }
}
