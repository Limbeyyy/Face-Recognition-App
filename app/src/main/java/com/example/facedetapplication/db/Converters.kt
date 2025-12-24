package com.example.facedetapplication.db
import androidx.room.TypeConverter
import java.nio.ByteBuffer
import java.nio.ByteOrder
class Converters {
    @TypeConverter
    fun fromFloatArray(arr: FloatArray): ByteArray {
        val bb = ByteBuffer.allocate(arr.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        arr.forEach { bb.putFloat(it) }
        return bb.array()
    }

    @TypeConverter
    fun toFloatArray(bytes: ByteArray): FloatArray {
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(bytes.size / 4) { bb.float }
    }
}
