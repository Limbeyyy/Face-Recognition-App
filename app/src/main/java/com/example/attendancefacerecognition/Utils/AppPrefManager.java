package com.example.facedetproject.Utils;

import android.content.Context;
import android.content.SharedPreferences;

public class AppPrefManager {
    private static final String PREF = "app_pref";
    private static final String KEY_DEVICE = "device_id";
    private static final String KEY_PASS = "password";
    private static final String KEY_URL = "url";

    private SharedPreferences prefs;
    public AppPrefManager(Context ctx){ prefs = ctx.getSharedPreferences(PREF, Context.MODE_PRIVATE); }

    public void setDeviceId(String id){ prefs.edit().putString(KEY_DEVICE, id).apply(); }
    public String getDeviceId(){ return prefs.getString(KEY_DEVICE, ""); }

    public void setPassword(String p){ prefs.edit().putString(KEY_PASS, p).apply(); }
    public String getPassword(){ return prefs.getString(KEY_PASS, ""); }

    public void setUrl(String url){ prefs.edit().putString(KEY_URL, url).apply(); }
    public String getUrl(){ return prefs.getString(KEY_URL, ""); }
}
