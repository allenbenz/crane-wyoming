# Using crane-wyoming with Home Assistant

This assumes `crane-wyoming` is already running, either as a [systemd
service](../README.md#running-as-a-systemd-service) or via the
[ha-crane-wyoming](https://github.com/cryptomilk/ha-crane-wyoming) Home
Assistant App, and is reachable on its Wyoming port (default `10200`).

## Adding the integration

- If installed via the ha-crane-wyoming App, the App Store's supervisor
  advertises the service (`discovery: wyoming`) and Home Assistant should offer
  it under **Settings -> Devices & Services -> Discovered**; accept it there.
- Otherwise (systemd service, or discovery didn't trigger), add it manually:
  **Settings -> Devices & Services -> Add Integration -> Wyoming Protocol**,
  then enter the host/IP and port where `crane-wyoming` is listening (default
  `10200`).

This creates one text-to-speech entity (and, if an ASR model is loaded, one
speech-to-text entity) for that `crane-wyoming` instance.

## Setting up an Assist voice assistant

**Settings -> Voice assistants -> Add assistant** builds a pipeline that ties a
conversation agent, speech-to-text, and text-to-speech engine together.

1. Under **Configuration**, name the assistant and pick a language.
2. Under **Speech-to-text**, pick the crane-wyoming entity if you have an ASR
   model loaded. Leave it `None` otherwise.
3. Under **Text-to-speech**, pick the crane-wyoming entity. Home Assistant then
   adds a **Voice** field for it, populated from the voices crane-wyoming
   advertises.
4. Pick the voice you want, e.g. `serena`.

This Voice field is a real dropdown, unlike the `tts.speak` Options field
below. It's populated the same way, so the same caveat applies: see ["Only one
model's voices are advertised to Home
Assistant"](#only-one-models-voices-are-advertised-to-home-assistant) if the
voice you want isn't listed.

## Testing with a Speak action

**Developer Tools -> Actions**, search for **Text-to-speech: Speak**
(`tts.speak`). Fill in the TTS entity created above, a media player entity,
your message, and a language (e.g. `de` for German), then run the action.

### Selecting a specific voice (e.g. "serena" for Qwen3 CustomVoice)

crane-wyoming reports every voice a loaded model defines (e.g. a Qwen3-TTS
CustomVoice model's predefined speakers) as its own Wyoming *voice*. There is
no separate "speaker" concept to configure.

Type the voice name directly into the Speak action's **Options** field:

```yaml
voice: serena
```

Voice names are exact and case-sensitive. For `Qwen3-TTS-12Hz-0.6B-CustomVoice`
they come straight from the model's `spk_id` keys: `serena`, `vivian`,
`uncle_fu`, `ryan`, `aiden`, `ono_anna`, `sohee`, `eric`, `dylan`. Note the
lowercase, not `Serena`. Run `cw-say --list-voices --uri <your-uri>` against a
running server to get the exact names for whatever models you have loaded.

If you'd rather edit the whole action call at once, use the action editor's
&#8942; menu -> "Edit in YAML":

```yaml
action: tts.speak
target:
  entity_id: tts.crane_wyoming
data:
  media_player_entity_id: media_player.xxx
  message: "Hallo, ich bin Serena"
  language: de
  options:
    voice: serena
```

## Only one model's voices are advertised to Home Assistant

Home Assistant reads the list of available voices per language from
`crane-wyoming`'s `describe` response, but only from its *first* TTS model.
`crane-wyoming` always sorts loaded models alphabetically by directory name
there, regardless of `--model-tts` order. Say you load a voice-cloning model
(`Qwen3-TTS-12Hz-0.6B-Base`) alongside a CustomVoice model
(`Qwen3-TTS-12Hz-0.6B-CustomVoice`). Home Assistant only ever learns about the
alphabetically-first one's voices. `...-Base` sorts before `...-CustomVoice`
and defines no named voices at all. So with both loaded, Home Assistant would
think this TTS entity has no voices or languages.

This doesn't block typing `voice: serena` into Options by hand, as above.
`crane-wyoming` resolves it against every model it has loaded, not just the one
it happens to advertise as "first". It only matters if you rely on Home
Assistant surfacing voices itself elsewhere, e.g. a voice assistant pipeline's
TTS voice setting. To fix that, either:

- Restrict the instance Home Assistant talks to so it only loads the
  CustomVoice model. For the systemd service, set
  `CRANE_WYOMING_MODEL_TTS=Qwen3-TTS-12Hz-0.6B-CustomVoice` in
  `/etc/default/crane-wyoming` (or `~/.config/crane-wyoming` for the per-user
  unit) and restart the service. For the App, there is no equivalent option;
  only keep the one model you want under `/share/crane-wyoming/tts/`.
- Or, to use several TTS models at once, run separate `crane-wyoming` instances
  (different ports/sockets) and add a separate Wyoming integration per
  instance.
