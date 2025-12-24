package com.example.facedetapplication.db
import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
@Dao
interface FaceDao {

    @Insert
    fun insertPerson(person: PersonEntity): Long

    @Insert
    fun insertEmbedding(embedding: EmbeddingEntity)

    @Query("SELECT * FROM persons")
    fun getPersons(): List<PersonEntity>

    @Query("SELECT * FROM embeddings WHERE personId = :pid")
    fun getEmbeddings(pid: Long): List<EmbeddingEntity>
}
