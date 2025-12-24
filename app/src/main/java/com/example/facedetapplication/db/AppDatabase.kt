package com.example.facedetapplication.db
import androidx.room.Database
import androidx.room.TypeConverters
import androidx.room.RoomDatabase
import android.content.Context
import androidx.room.Room

@Database(
    entities = [PersonEntity::class, EmbeddingEntity::class],
    version = 1
)
@TypeConverters(Converters::class)
abstract class AppDatabase : RoomDatabase() {
    abstract fun dao(): FaceDao

    companion object {
        fun create(context: Context): AppDatabase =
            Room.databaseBuilder(
                context,
                AppDatabase::class.java,
                "faces.db"
            ).allowMainThreadQueries().build()
    }
}
