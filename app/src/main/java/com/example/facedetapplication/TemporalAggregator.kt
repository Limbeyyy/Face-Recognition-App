package com.example.facedetapplication

class TemporalAggregator {
    private val scores = mutableMapOf<String, Float>()
    private val decay = 0.85f

    fun update(name: String, confidence: Float): String {
        scores.keys.forEach { scores[it] = scores[it]!! * decay }
        scores[name] = (scores[name] ?: 0f) + confidence
        return scores.maxByOrNull { it.value }?.key ?: "Unknown"
    }
}
