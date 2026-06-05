import streamlit as st
import streamlit.components.v1 as components

def voice_input_component(key: str = "voice_transcript"):
    """Injects browser-native SpeechRecognition. Returns transcript via st.session_state."""
    if key not in st.session_state: st.session_state[key] = ""

    script = """
    <button id="micBtn" style="padding:10px 20px; font-size:16px; cursor:pointer;">🎙️ Speak</button>
    <div id="status" style="margin-top:5px; font-size:12px; color:#666;">Ready</div>
    <script>
    const btn = document.getElementById('micBtn');
    const status = document.getElementById('status');
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) { status.innerText = "STT not supported"; btn.disabled=true; }
    else {
        const rec = new SpeechRecognition();
        rec.lang = 'en-IN'; rec.continuous = false; rec.interimResults = false;
        rec.onstart = () => status.innerText = "🔴 Listening...";
        rec.onresult = (e) => {
            const t = e.results[0][0].transcript;
            window.parent.postMessage({type: 'streamlit:SET_SESSION_STATE', value: t, key: '%s'}, '*');
            status.innerText = "✅ Captured: " + t;
        };
        rec.onerror = () => status.innerText = "❌ Error. Try again.";
        rec.onend = () => status.innerText = "Ready";
        btn.onclick = () => rec.start();
    }
    </script>
    """ % key

    components.html(script, height=80)
    st.session_state[key] = st.text_input("🎙️ Voice Transcript (Paste or use mic above)", value=st.session_state[key])
    return st.session_state[key]