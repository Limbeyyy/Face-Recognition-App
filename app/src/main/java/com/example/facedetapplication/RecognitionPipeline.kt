package com.example.facedetapplication

import android.graphics.Bitmap
import com.example.facedetapplication.model.FaceBox
import com.example.facedetapplication.model.RecognitionResult
import com.example.facedetapplication.util.BlurEstimator
import kotlin.collections.maxBy


class RecognitionPipeline(
    private val detector: FaceDetector,
    private val landmark: LandmarkExtractor,
    private val embedder: FaceRecognition,
    private val spoof: AntiSpoof,
    private val storage: VectorStorage,
    private val temporal: TemporalAggregator
) {

    fun process(bitmap: Bitmap): RecognitionResult? {

        // 1. Detect faces
        val faces = detector.detect(bitmap)
        if (faces.isEmpty()) return null

        // 2. Take highest-confidence face
        val face: FaceBox = faces.maxBy { it.confidence }

        // 3. Anti-spoof
        val spoofScore = spoof.predict(bitmap, face.bbox)
        if (spoofScore < 0.5f) return null

        // 4. Landmarks + quality
        val landmarks = landmark.extract(bitmap, face.bbox)
        val landmarkWeight = landmark.quality(landmarks)

        // 5. Blur quality
        val blurWeight = BlurEstimator.score(bitmap, face.bbox)

        // 6. Combined weight
        val weight =
            face.confidence *
                    landmarkWeight *
                    blurWeight *
                    spoofScore

        // 7. Face embedding
        val rawEmbedding = embedder.embed(bitmap, face.bbox)

        val weightedEmbedding = FloatArray(rawEmbedding.size) { i ->
            rawEmbedding[i] * weight
        }

        // 8. Vector match
        val (name, similarity) = storage.match(weightedEmbedding)

        // 9. Temporal aggregation
        val finalName = temporal.update(name, similarity)

        return RecognitionResult(
            name = finalName,
            confidence = similarity
        )
    }
}
