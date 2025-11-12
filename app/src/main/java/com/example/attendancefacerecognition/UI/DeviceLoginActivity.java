package com.example.facedetproject.UI;

import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.example.facedetproject.MainActivity;
import com.example.facedetproject.R;
import com.example.facedetproject.Utils.AppPrefManager;

public class DeviceLoginActivity extends AppCompatActivity {
    private EditText etDeviceId, etPassword, etUrl;
    private Button btnConfirm;
    private TextView tvError;
    private AppPrefManager appPref;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.device_id_dialog_layout);

        appPref = new AppPrefManager(this);

        etDeviceId = findViewById(R.id.etDeviceId);
        etPassword = findViewById(R.id.etPassword);
        etUrl = findViewById(R.id.etUrl);
        btnConfirm = findViewById(R.id.btnConfirm);
        tvError = findViewById(R.id.tvError);

        btnConfirm.setOnClickListener(v -> {
            String d = etDeviceId.getText().toString().trim();
            String p = etPassword.getText().toString().trim();
            String u = etUrl.getText().toString().trim();

            if (d.isEmpty()) {
                tvError.setText("Enter Device ID"); tvError.setTextColor(Color.RED); return;
            }
            if (p.isEmpty()) {
                tvError.setText("Enter Password"); tvError.setTextColor(Color.RED); return;
            }
            if (u.isEmpty()) {
                tvError.setText("Enter URL"); tvError.setTextColor(Color.RED); return;
            }

            appPref.setDeviceId(d); appPref.setPassword(p); appPref.setUrl(u);
            startActivity(new Intent(this, MainActivity.class));
            finish();
        });
    }
}
