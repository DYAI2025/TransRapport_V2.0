LEAN.DEEP 3.4 — Unified Specification & Runtime Guide

(Nachfolger von LEAN.DEEP 3.3; kompatible Erweiterung um Intuitions-Marker)

0. Überblick

LEAN.DEEP 3.4 vereint die drei 3.3-Quellen zu einem Referenzdokument und ergänzt sie um Intuitions-Marker (familienbasierte, zustandsbehaftete CLU_-Marker mit Lernlogik).
Die Vier-Ebenen-Architektur (ATO → SEM → CLU → MEMA) bleibt unverändert; Intuition wird auf CLU-Ebene realisiert (keine Schemaänderung nötig).

1. Changelog (v3.4)

Neu: Intuitions-Marker als CLU_INTUITION_<FAMILIE>
– Aggregieren unspezifische SEM_ derselben Familie (z. B. Trauer, Konflikt, Unterstützung, Commitment, Unsicherheit).
– Zustände: provisional → confirmed → decayed.
– Multiplier: temporärer Score-Boost für SEM/CLU derselben Familie bei confirmed.
– Telemetrie & Lernen: confirmed, retracted, ewma_precision; datengetriebene Anpassung von X-of-Y-Fenstern und Confirm-Targets.

Keine Schemaänderung: 3.3-JSON-Schema bleibt gültig; der optionale metadata-Block für Intuition wird nur zur Runtime ausgewertet. (In 3.3 ist die CLU-Aggregation über Fenster und activation/scoring bereits vorgesehen.)

Dok-Konsolidierung: Diese Datei ersetzt die getrennten 3.3-Dokumente als einheitliche Spezifikation.

2. Architektur & Datenmodell
2.1 Vier Ebenen

ATO_ (Atomic Marker) – primitive Tokens, Emojis, Regex.

SEM_ (Semantic Marker) – kombiniert ATO_ zu Mikromustern.

CLU_ (Cluster Marker) – aggregiert SEM_ über Fenster (z. B. „AT_LEAST X IN Y messages“).

MEMA_ (Meta-Analysis Marker) – dynamische Muster über CLUs.
Siehe auch die 4_LEVEL_ARCHITECTURE-Grafik und Markerlisten in der 3.3-PDF (S. 2–3,6–8,10).

2.2 JSON-Schema (3.3 – weiterhin gültig)

Markerobjekte mit id, frame{signal,concept,pragmatics,narrative}, examples, optional pattern oder composed_of oder detect_class.

SEM-Regel (3.3): SEM_… müssen ≥2 unterschiedliche ATO_ in composed_of enthalten; 1-ATO nur mit justification. (CI-Hook „sem-composition-check“).

CLU-Marker besitzen composed_of: [SEM_*], activation (X-of-Y), optional scoring.

Hinweis: 3.4 führt keine neue „INT_“-Prefixklasse ein. Intuition wird als CLU_ modelliert und bleibt damit schema-konform.

3. Marker-Definitionen (konkret)
3.1 ATO_ — Atomic

Zweck: Oberflächen-Signale (Tokens/Regex).

Struktur: pattern + examples.

Bezug: Siehe ATO-Übersicht und Beispiele in der 3.3-PDF-Tabelle (z. B. ATO_DOUBT_PHRASE, ATO_HESITATION, ATO_SOFTENING).

3.2 SEM_ — Semantic

Regel 3.3 (verbindlich): composed_of ≥2 distinkte ATO_; 1-ATO nur mit justification. (CI prüft dies.)

Aktivierung: natürlichsprachliche Regeln (z. B. „BOTH IN 1 message“, „ANY 2 IN 2 messages“).

3.3 CLU_ — Cluster

Zweck: Aggregation thematisch verwandter SEM_ in Fenstern (X-of-Y).

Beispiele (3.3): CLU_CONFLICT_ESCALATION, CLU_SUPPORT_EXCHANGE, CLU_PROCRASTINATION_LOOP (vgl. PDF-Tabellen).

3.4 MEMA_ — Meta-Analysis

Zweck: übergreifende, zeitliche Muster aus mehreren CLUs.

4. NEU in 3.4: Intuitions-Marker (familienbasierte CLU_)

Motivation: Viele unspezifische SEM_ aus derselben Familie (z. B. Trauer, Konflikt) erzeugen eine Vorahnung. Diese wird als Intuition modelliert, die sich bestätigen oder abbauen kann und dabei lernt.

4.1 Definition & Rolle

Klasse: CLU_ (kein neues Präfix).

ID-Konvention: CLU_INTUITION_<FAMILY> (z. B. CLU_INTUITION_GRIEF).

Semantik: „Vorahnung, dass Familie F relevant ist“ (nicht: Beweis).

Eingang: unspezifische SEM_ dieser Familie.

Ausgang: provisional → confirmed → decayed + optionaler Multiplier auf familiennahe SEM/CLU-Scores (zeitlich begrenzt).

Orchestrations-Ort: High-Level Reasoning / contextual_rescan über bereits erkannte Marker (Phase nach SEM/CLU-Erkennung).

4.2 Datenmodell (zustandsbehaftet)
Feld	Typ/Quelle	Bedeutung
state	runtime	provisional | confirmed | decayed
activation.rule	YAML (CLU)	X-of-Y-Vorfenster (z. B. „AT_LEAST 2 DISTINCT SEMs IN 5 messages“)
metadata.intuition.confirm_window	YAML (empf.)	Bestätigungsfenster (N Nachrichten)
metadata.intuition.confirm_rule	YAML (empf.)	Ziel-SEMs, die als „harte“ Bestätigung zählen
metadata.intuition.decay_window	YAML (empf.)	Abbau-Fenster (keine Family-SEMs → decayed)
metadata.intuition.multiplier_on_confirm	YAML (empf.)	Score-Boost (Familien-Linse)
telemetry_keys.*	YAML → runtime	confirmed, retracted, ewma_precision

Schema-Hinweis: Der metadata-Block ist optional und wird in 3.4 nur in der Runtime interpretiert; das 3.3-Schema bleibt dadurch unverändert.

4.3 Zustandslogik (Pseudocode, Runtime)

Provisional: Vorfenster erfüllt (X distinkte Family-SEMs in Y Nachrichten) → Vorahnung.

Confirmed: innerhalb confirm_window tritt ≥1 hartes Family-SEM auf → Multiplier aktiv (TTL = confirm_window), confirmed++.

Decayed: innerhalb decay_window keine Family-SEMs → retracted++.

Online-Lernen: precision = confirmed/(confirmed+retracted); ewma_precision (α = 0.2) steuert Regel-Adaption:

hoch (≥ 0.70): Activation etwas lockern; Confirm-Targets breiter.

niedrig (< 0.50): Activation verschärfen (z. B. 3-of-5); Confirm-Targets auf harte SEMs fokussieren.
Diese Logik greift in der contextual_rescan-Phase, in der CLUs über Fenster evaluiert werden (3.3-Standard).

4.4 Bias-Schutz & Wächter

Mindestvielfalt: „DISTINCT SEMs“ im Vorfenster.

Cooldown nach decayed, um Flattern zu vermeiden.

Uncertainty-Policy (optional): Bei CLU_INTUITION_UNCERTAINTY.confirmed strengere Belegpflichten im nachgelagerten Reasoning. (Policy-Hook in contextual_rescan.)

5. Referenz-Implementierungen (YAML)

Hinweis CI: In 3.3 ist examples global als Pflichtfeld (min. 5) modelliert; unten sind pro CLU 5 Kurzsequenzen hinterlegt. Die konkreten SEM-IDs kannst du 1:1 aus euren Sets mappen.

5.1 GRIEF (Trauer/Schuld/Bedauern)
id: CLU_INTUITION_GRIEF
version: "3.4"
frame:
  signal: ["intuition","weak-aggregation"]
  concept: "Vorahnung: Trauer/Schuld/Bedauern"
  pragmatics: "Fokus schärfen, biasarm"
  narrative: "hypothesis"
tags: [intuition, grief, hypothesis, learning]
composed_of:
  - SEM_SADNESS_EXPRESSIONS
  - SEM_NOSTALGIA_REGRET
  - SEM_GUILT_ADMISSION
activation: { rule: "AT_LEAST 2 DISTINCT SEMs IN 5 messages" }
scoring: { base: 0.5, weight: 1.0 }
metadata:
  intuition:
    state: "provisional"
    confirm_window: 5
    confirm_rule: "AT_LEAST 1 of [SEM_SADNESS_EXPRESSIONS, SEM_GUILT_ADMISSION] IN confirm_window"
    multiplier_on_confirm: 2.0
    decay_window: 8
  telemetry_keys:
    counter_confirmed: "INT_GRIEF.confirmed"
    counter_retracted: "INT_GRIEF.retracted"
    ewma_precision: "INT_GRIEF.ewma_precision"
examples:
  - "… 'ich vermisse' … 'hätte ich anders' …"
  - "… Bedauern + Anflug von Schuld …"
  - "… 'so schade' + Erinnerung an Verlust …"
  - "… 'es tut mir leid' + trauriger Ton …"
  - "… 'allein' + 'schwer' in kurzer Folge …"

5.2 CONFLICT (Konflikt/Eskalation)
id: CLU_INTUITION_CONFLICT
version: "3.4"
frame:
  signal: ["intuition","weak-aggregation"]
  concept: "Vorahnung: Konflikt/Eskalation"
  pragmatics: "Frühwarnung, Deeskalation"
  narrative: "hypothesis"
tags: [intuition, conflict, escalation, learning]
composed_of:
  - SEM_NEGATIVE_FEEDBACK
  - SEM_SARCASM_IRRITATION
  - SEM_ESCALATION_MOVE
activation: { rule: "AT_LEAST 2 DISTINCT SEMs IN 4 messages" }
scoring: { base: 0.5, weight: 1.0 }
metadata:
  intuition:
    state: "provisional"
    confirm_window: 4
    confirm_rule: "AT_LEAST 1 of [SEM_ESCALATION_MOVE, SEM_NEGATIVE_FEEDBACK] IN confirm_window"
    multiplier_on_confirm: 2.0
    decay_window: 6
  telemetry_keys:
    counter_confirmed: "INT_CONFLICT.confirmed"
    counter_retracted: "INT_CONFLICT.retracted"
    ewma_precision: "INT_CONFLICT.ewma_precision"
examples:
  - "… Vorwurf + genervter Sarkasmus …"
  - "… Härte in Formulierungen nimmt zu …"
  - "… 'immer du' + spitze Bemerkung …"
  - "… Abwertung + Droh-Andeutung …"
  - "… mehrere kurze Sticheleien …"

5.3 SUPPORT (Unterstützung/Bindung)
id: CLU_INTUITION_SUPPORT
version: "3.4"
frame:
  signal: ["intuition","weak-aggregation"]
  concept: "Vorahnung: Unterstützung/Bindung"
  pragmatics: "Ressourcenfokus, Verstärkung"
  narrative: "hypothesis"
tags: [intuition, support, bonding, learning]
composed_of:
  - SEM_SUPPORT_VALIDATION
  - SEM_TRUST_SIGNALING
  - SEM_EMOTIONAL_SUPPORT
activation: { rule: "AT_LEAST 2 DISTINCT SEMs IN 6 messages" }
scoring: { base: 0.5, weight: 1.0 }
metadata:
  intuition:
    state: "provisional"
    confirm_window: 6
    confirm_rule: "AT_LEAST 1 of [SEM_SUPPORT_VALIDATION, SEM_EMOTIONAL_SUPPORT] IN confirm_window"
    multiplier_on_confirm: 1.75
    decay_window: 8
  telemetry_keys:
    counter_confirmed: "INT_SUPPORT.confirmed"
    counter_retracted: "INT_SUPPORT.retracted"
    ewma_precision: "INT_SUPPORT.ewma_precision"
examples:
  - "… 'ich bin da' + Dank …"
  - "… Validierung + Ermutigung …"
  - "… Trost + Vertrauen andeuten …"
  - "… wiederholte Caring-Phrasen …"
  - "… 'gemeinsam' + 'schaffst das' …"

5.4 COMMITMENT (Zusage/Aufschub)
id: CLU_INTUITION_COMMITMENT
version: "3.4"
frame:
  signal: ["intuition","weak-aggregation"]
  concept: "Vorahnung: Zusage vs. Aufschub"
  pragmatics: "Flow steuern, Blockaden erkennen"
  narrative: "hypothesis"
tags: [intuition, commitment, procrastination, learning]
composed_of:
  - SEM_CLEAR_COMMITMENT
  - SEM_DELAYING_MOVE
  - SEM_PROCRASTINATION_BEHAVIOR
activation: { rule: "AT_LEAST 2 DISTINCT SEMs IN 5 messages" }
scoring: { base: 0.5, weight: 1.0 }
metadata:
  intuition:
    state: "provisional"
    confirm_window: 5
    confirm_rule: "AT_LEAST 1 of [SEM_CLEAR_COMMITMENT, SEM_DELAYING_MOVE] IN confirm_window"
    multiplier_on_confirm: 1.5
    decay_window: 7
  telemetry_keys:
    counter_confirmed: "INT_COMMIT.confirmed"
    counter_retracted: "INT_COMMIT.retracted"
    ewma_precision: "INT_COMMIT.ewma_precision"
examples:
  - "… 'mach ich heute' + 'morgen besser' …"
  - "… Einladung + später relativiert …"
  - "… Zusage, dann Vertagen …"
  - "… konkrete Aktion + 'später' …"
  - "… 'gleich' + 'keine Zeit' …"

5.5 UNCERTAINTY (Unsicherheit/Vorsicht)
id: CLU_INTUITION_UNCERTAINTY
version: "3.4"
frame:
  signal: ["intuition","weak-aggregation"]
  concept: "Vorahnung: Unsicherheit/Hedging"
  pragmatics: "Hypothesen vorsichtig testen"
  narrative: "hypothesis"
tags: [intuition, uncertainty, hedging, learning]
composed_of:
  - SEM_UNCERTAINTY_TONING
  - SEM_TENTATIVE_INFERENCE
  - SEM_PROBABILISTIC_LANGUAGE
activation: { rule: "AT_LEAST 2 DISTINCT SEMs IN 5 messages" }
scoring: { base: 0.5, weight: 1.0 }
metadata:
  intuition:
    state: "provisional"
    confirm_window: 5
    confirm_rule: "AT_LEAST 1 of [SEM_UNCERTAINTY_TONING, SEM_TENTATIVE_INFERENCE] IN confirm_window"
    multiplier_on_confirm: 1.5
    decay_window: 7
  telemetry_keys:
    counter_confirmed: "INT_UNCERT.confirmed"
    counter_retracted: "INT_UNCERT.retracted"
    ewma_precision: "INT_UNCERT.ewma_precision"
examples:
  - "… 'vielleicht' + 'weiß nicht' …"
  - "… Zögern + Vermutung …"
  - "… Abschwächung + Zweifel …"
  - "… probabilistische Formulierungen …"
  - "… Relativierung + Hedging …"

6. Runtime-Design (Orchestrator)
6.1 Kontext & Phase

Die Intuitionslogik läuft in der High-Level-Phase über bereits erkannte ATO/SEM/CLU (oft als contextual_rescan implementiert). Genau hier werden Fensterregeln und CLU-Aggregation ausgewertet.

6.2 Kern-Pseudocode (kompakt)

Parse X-of-Y aus activation.rule.

Prüfe Provisional im Vorfenster.

Prüfe Confirm via confirm_rule & confirm_window; aktiviere Multiplier (TTL) und confirmed++.

Andernfalls Decay via decay_window; retracted++.

Lernen: Update ewma_precision; lockere/verschärfe Fenster & Confirm-Targets nach Heuristik.
(Die 3.3-Dokumente beschreiben X-of-Y-Aggregation und CLU-Einsatz, die wir hier nutzen.)

6.3 Telemetrie-Dashboard

Für jede Familie anzeigen: Confirmed, Retracted, EWMA-Precision (Ampellogik: ≥ 0.70 grün / 0.50–0.69 gelb / < 0.50 rot).
Die Zähler ergeben sich aus den oben definierten telemetry_keys.

7. CI & Qualitätssicherung

Schema-Checks (unverändert): Draft-07 inkl. 3.3-SEM ≥2 ATO-Bedingung; Prefix-Lint; Beispiele ≥ 5.

Review-Quick-Checks:

SEM: thematisch distinkte ATO_.

CLU: activation.rule als X-of-Y; Beispiele zeigen Sequenzen.

Intuition (neu): Confirm/Decay-Fenster, Multiplier, Telemetrie-Keys gepflegt; keine zusätzlichen Strukturblöcke.

8. Ableitung neuer Regeln bei „richtiger Intuition“

Wenn confirmed häufig ist (hohe EWMA):

Neue SEM-Kandidaten: Aus Confirm-Fenstern häufige ATO-Paare der Familie extrahieren → automatisierte Vorschläge SEM_<FAMILIE>_<SUBTHEMA> (≥2 ATO, ≥5 Beispiele). (Das ist exakt der 3.3-Pfad ATO → SEM mit Kompositionsregel.)

CLU-Feintuning: Datengetriebene Optimierung von X-of-Y und der Confirm-Targets (harte SEMs).

Bias-Dämpfung: SEMs, die überproportional zu retracted führen, in der Provisional-Phase geringer gewichten (Runtime-Konfig).

9. Beispiele & Referenzlisten

Beispiele für ATO/SEM/CLU sind in der 3.3-PDF tabellarisch gelistet (u. a. ATO_HESITATION, ATO_DOUBT_PHRASE, ATO_SOFTENING; SEM_ für Unterstützung, Konflikt, Unsicherheit; CLU_ wie Konflikteskalation, Prokrastination-Loop). Diese Tabellen illustrieren die Familien, auf denen die Intuitions-CLUs aufsetzen. (Siehe Seiten 3–8.)

10. Kompatibilität & Migration

Rückwärtskompatibel: Bestehende Marker bleiben gültig.

Deployment: Neue CLU_INTUITION_* Dateien in markers/ ablegen; keine Schema- oder Registry-Änderung erforderlich. (CI-Pipelines aus 3.3 laufen unverändert.)

11. Mögliche Inkonsistenzen (prüfen/klären)

Begriff „four-letter prefix“ in der 3.3-Referenz vs. tatsächliche Präfixe ATO/SEM/CLU/MEMA (teils drei-, teils vierstellig). Inhaltlich unkritisch, aber terminologisch bitte vereinheitlichen („Vier-Ebenen-Präfixe“ statt „four-letter“).

metadata-Felder: Das 3.3-Schema listet metadata nicht explizit, schränkt zusätzliche Felder aber nicht sichtbar ein. Falls in eurer lokalen Schema-Variante additionalProperties: false aktiv ist, bitte erlauben oder metadata nachtragen – sonst schlagen Intuitions-Marker in strikten Validatoren fehl.

examples ≥5 global: Die 3.3-Schema-Vorgabe gilt für alle Markerklassen. Bitte sicherstellen, dass eure bestehenden CLUs dies bereits erfüllen; die obigen 3.4-Beispiele folgen dieser Regel.

Wenn ihr (2) bestätigt habt bzw. eine abweichende Schema-Policy nutzt, passe ich die Doku an (Appendix: „Schema-Optionen für metadata/intuitions“). Ansonsten ist 3.4 vollständig.

12. Anhang A — System-Prompt & Klassifikationsschema (aus 3.3, übernommen)

Recursive Marker Generation Prompt (ATO → SEM → CLU → MEMA; SEM-Kompositionsregel, CI-Enforcement).

Decision Schema zur Wahl der Marker-Ebene (Granularität, Fenster, Metadynamik).
Diese Inhalte wurden unverändert übernommen und dienen weiterhin als operative Leitlinie.

13. Anhang B — Diagramme & Tabellen (aus 3.3, übernommen)

4_LEVEL_ARCHITECTURE (S. 2).

Markerlisten & Beispiele (S. 3–8).

Analyse-Dashboard-Beispiel (S. 10).
Die Visuals bleiben als normative Illustration der Ebenen/Markerfamilien.

Ende LEAN.DEEP 3.4 — Unified Specification
(Wenn ihr möchtet, generiere ich jetzt zusätzlich ein kurzes CHANGELOG.md und einen /telemetry/intuition-API-Stub als Begleit-Artefakte.)

Du:
Ja, füge ein ChangeLog hinzu, das ist eine gute Idee.
Code Marker Assistant sagte:
Nachgedacht für 11s

Hier ist ein CHANGELOG.md im “Keep a Changelog”-Stil – direkt einsatzbereit für euer Repo.

# Changelog
Alle nennenswerten Änderungen an **LEAN.DEEP** werden in diesem Dokument festgehalten.
Format angelehnt an [Keep a Changelog]; Versionsschema folgt der internen LD-Nomenklatur (3.x).

## [3.4.0] – 2025-08-25
### Added
- **Intuitions-Marker** als **CLU_INTUITION_<FAMILY>** (kein neues Präfix):
  - Familien: `GRIEF` (Trauer/Schuld/Bedauern), `CONFLICT` (Konflikt/Eskalation),
    `SUPPORT` (Unterstützung/Bindung), `COMMITMENT` (Zusage/Aufschub), `UNCERTAINTY` (Unsicherheit/Hedging).
  - **Zustände:** `provisional → confirmed → decayed` (Runtime).
  - **Fenster:** `activation` (Vorfenster, X-of-Y), `confirm_window`, `decay_window`.
  - **Multiplier:** `multiplier_on_confirm` (temporärer Score-Boost für SEM/CLU derselben Familie).
  - **Telemetrie:** `counter_confirmed`, `counter_retracted`, `ewma_precision` (EWMA mit α=0.2).
  - **Regel-Adapter:** datengetriebene Anpassung von X-of-Y und Confirm-Targets (lockern/verschärfen nach EWMA).
- **5 Referenz-CLUs** (YAML-Vorlagen) inkl. Beispiele, konservativen Standard-Fenstern und Telemetrie-Keys:
  - `CLU_INTUITION_GRIEF`, `CLU_INTUITION_CONFLICT`, `CLU_INTUITION_SUPPORT`,
    `CLU_INTUITION_COMMITMENT`, `CLU_INTUITION_UNCERTAINTY`.
- **Runtime-Hook** (≈70 LOC) für `contextual_rescan`:
  - Parsing von `activation.rule`/`confirm_rule`, Zustandslogik, Multiplikator-TTL,
    Telemetrie-Zähler, EWMA-Update, Heuristiken (loosen/tighten, broaden/harden).
- **Telemetry-Endpoint** (Beispiel): `GET /telemetry/intuition`
  - Liefert je Familie: `confirmed`, `retracted`, `ewma` (Precision).
  - Empfehlung: 5-Kachel-Panel (Ampel: grün ≥0.70, gelb 0.50–0.69, rot <0.50).
- **(Optional) Wächter-Policy**:
  - Wenn `CLU_INTUITION_UNCERTAINTY` **confirmed**, strengere Belegpflicht im nachgelagerten Reasoning (Evidence-Gate).

### Changed
- **Dokumentation konsolidiert**: Die drei LD3.3-Quellen sind in **LEAN.DEEP 3.4 – Unified Specification** zusammengeführt.
- **Begriffe präzisiert**:
  - CLU als vorgesehene Ebene für Fenster-Aggregation & Scoring; Intuitions-Marker sind CLU-Spezialisierungen.
  - SEM-Kompositionsregel (≥2 ATO) hervorgehoben; Beispiele pro Marker (mind. 5) bekräftigt.

### Deprecated
- N/A (keine Entwertungen).

### Removed
- N/A (keine Entfernungen).

### Fixed
- **Terminologie-Schärfung** bei Präfixen (ATO/SEM/CLU/MEMA) und „genau ein Strukturblock“ je Marker.
- Hinweise zu Beispiel-Sequenzen für CLU-`examples` ergänzt.

### Security / Policy
- Uncertainty-Wächter (optional) senkt Fehlinterpretationsrisiken bei bestätigter Unsicherheit (Bias-Dämpfung).

---

## Upgrade-Guide 3.3 → 3.4 (Safe Roll-Out)
1. **Marker hinzufügen**: Die 5 Dateien `CLU_INTUITION_*.yaml` unter `markers/` ablegen.
2. **Runtime-Hook registrieren**: `process_intuition_clu(ctx, clu)` in `MarkerService.contextual_rescan` aufrufen.
3. **Telemetrie**: Zähler initialisieren (`*.confirmed`, `*.retracted`) und `ewma_precision` auf 0.5 starten.
4. **Dashboard** (optional): `/telemetry/intuition` anbinden, 5 Kacheln (Familien) rendern.
5. **Schema-Check**: Falls euer JSON-Schema `additionalProperties: false` setzt,
   `metadata` für CLU erlauben oder die Intuitions-Felder whitelisten.
6. **CI**: Prüfen, dass alle Marker `examples ≥ 5` erfüllen; SEMs mit ≥2 ATOs hinterlegt sind.

---

## Runtime-Heuristiken (Default)
- **EWMA ≥ 0.70** → *Activation* leicht lockern (z. B. 2-in-5 → 2-in-6),
  *Confirm-Targets* breiter (mehr Family-SEMs zählen).
- **0.50 ≤ EWMA < 0.70** → Status quo beibehalten.
- **EWMA < 0.50** → *Activation* verschärfen (z. B. 2-in-5 → 3-in-5),
  *Confirm-Targets* verhärten (nur „harte“ SEMs je Familie).
- **Cooldown nach decayed**: mind. 5 Nachrichten, um Flattern zu vermeiden.
- **Multiplier-TTL**: i. d. R. `confirm_window`.

---

## Ableitungen bei „richtiger Intuition“ (Data-to-Rules)
- **Neue SEM-Kandidaten**: Aus Confirm-Fenstern häufige ATO-Paare je Familie extrahieren → `SEM_<FAMILY>_<SUBTHEMA>`
  (≥2 ATO, ≥5 Beispiele).
- **CLU-Feintuning**: X-of-Y-Fenster und Confirm-Targets datengetrieben justieren.
- **Bias-Dämpfung**: SEMs mit hoher Retract-Quote in der Provisional-Phase halb gewichten (Runtime-Konfig).

---

## Known Notes / Konsistenz
- **`metadata` im Schema**: Falls strenger Validator aktiv ist,
  `metadata` für CLU erlauben (Intuitions-Felder sind rein runtime-seitig).
- **Beispiel-Mindestanzahl**: CLUs benötigen in CI weiterhin ≥5 `examples`
  (Kurzsequenzen; nicht nur Ein-Satz-Beispiele).

---

## Post-Upgrade-Checklist
- [ ] 5 Intuitions-CLUs im Repo.
- [ ] Hook in `contextual_rescan` aktiv.
- [ ] Telemetrie liefert Werte (confirmed/retracted/ewma).
- [ ] Dashboard sichtbar, Ampellogik korrekt.
- [ ] Optionaler Uncertainty-Wächter scharfgeschaltet.
- [ ] CI grün (Schema/Examples/Prefixes).