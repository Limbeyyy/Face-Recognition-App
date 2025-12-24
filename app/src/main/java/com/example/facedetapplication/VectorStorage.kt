package com.example.facedetapplication

import com.example.facedetapplication.db.AppDatabase
import com.example.facedetapplication.db.PersonEntity
import com.example.facedetapplication.db.EmbeddingEntity
import kotlin.math.sqrt
class VectorStorage(private val db: AppDatabase) {

    fun add(person: String, embedding: FloatArray, weight: Float) {
        val dao = db.dao()
        val pid = dao.getPersons().find { it.name == person }
            ?.id ?: dao.insertPerson(PersonEntity(name = person))

        dao.insertEmbedding(
            EmbeddingEntity(
                personId = pid,
                vector = embedding,
                weight = weight
            )
        )
    }

    fun match(query: FloatArray, topKRatio: Float = 0.5f): Pair<String, Float> {
        var bestName = "Unknown"
        var bestScore = -1f

        val dao = db.dao()
        dao.getPersons().forEach { person ->
            val embs = dao.getEmbeddings(person.id)
            if (embs.isEmpty()) return@forEach

            val k = (embs.size * topKRatio).toInt().coerceAtLeast(1)
            val top = embs.sortedByDescending {
                cosine(it.vector, mean(embs.map { e -> e.vector }))
            }.take(k)

            val score = top.map { cosine(it.vector, query) }.average().toFloat()
            if (score > bestScore) {
                bestScore = score
                bestName = person.name
            }
        }
        return bestName to bestScore
    }

    private fun mean(vs: List<FloatArray>): FloatArray {
        val m = FloatArray(vs[0].size)
        vs.forEach { v -> for (i in v.indices) m[i] += v[i] }
        for (i in m.indices) m[i] /= vs.size
        return m
    }

    private fun cosine(a: FloatArray, b: FloatArray): Float {
        var d = 0f; var na = 0f; var nb = 0f
        for (i in a.indices) {
            d += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        return d / (sqrt(na) * sqrt(nb) + 1e-6f)
    }
}
