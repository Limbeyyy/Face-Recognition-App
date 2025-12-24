package com.example.facedetapplication

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

class FaceListAdapter(private val faces: List<String>) :
    RecyclerView.Adapter<FaceListAdapter.FaceViewHolder>() {

    class FaceViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val nameText: TextView = itemView.findViewById(R.id.face_name)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): FaceViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_face, parent, false)
        return FaceViewHolder(view)
    }

    override fun onBindViewHolder(holder: FaceViewHolder, position: Int) {
        holder.nameText.text = faces[position]
    }

    override fun getItemCount(): Int = faces.size
}
