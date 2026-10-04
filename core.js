// هسته پاسخ‌گویی نسخه مرورگری (Client-side)
//
// چرا این فایل وجود دارد؟ برای اینکه نسخه نمایشی روی Cloudflare Pages (که فقط فایل
// ایستا سرو می‌کند) بدون سرور کار کند. اعداد TF-IDF از هسته پایتون صادر شده‌اند
// (scripts/export_static_data.py) تا امتیازها با سرور اصلی یکسان بماند.
//
// تفاوت‌های آگاهانه با نسخه سرور:
//   * صدا با Web Speech API مرورگر تولید می‌شود، نه موتور عصبی edge-tts.
//   * لایه LLM اختیاری اینجا فعال نیست (کلید API نباید در مرورگر قرار بگیرد).
//   * جلسه گفت‌وگو فقط در حافظه همین تب نگه داشته می‌شود.

(function (global) {
  'use strict';

  const D = global.INIF_DATA;
  const ZWNJ = '\u200c';

  // ---------- نرمال‌سازی (هم‌راستا با app/core/normalize.py) ----------
  const CHAR_MAP = { 'ي': 'ی', 'ی': 'ی', 'ك': 'ک', 'ک': 'ک', 'ة': 'ه', 'ۀ': 'ه', 'أ': 'ا', 'إ': 'ا', 'ؤ': 'و' };
  const PUNCT_MAP = { '،': ',', '؛': ';', '؟': '?', 'ـ': '' };
  const FA = '۰۱۲۳۴۵۶۷۸۹', AR = '٠١٢٣٤٥٦٧٨٩';
  const DIACRITICS = '\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670\u0653\u0654\u0655';
  const BIDI = '\u200d\u200e\u200f\u200b\u202a\u202b\u202c\u202d\u202e';

  function stripBidi(t) {
    let out = '';
    for (const ch of t) if (!BIDI.includes(ch)) out += ch;
    return out;
  }

  function normalize(text, keepZwnj = true) {
    let t = String(text || '');
    // Python normalize() از NFC استفاده می‌کند و TfidfVectorizer هم strip_accents=None
    // دارد؛ یعنی «آ» (U+0622) دست‌نخورده می‌ماند. تجزیه NFD اینجا «آ» را به «ا»
    // تبدیل می‌کرد و n-gram‌ها واگرا می‌شدند (باگ گزارش‌شده).
    t = t.normalize('NFC');
    t = stripBidi(t);
    let out = '';
    for (const ch of t) {
      if (CHAR_MAP[ch] !== undefined) out += CHAR_MAP[ch];
      else if (PUNCT_MAP[ch] !== undefined) out += PUNCT_MAP[ch];
      else out += ch;
    }
    t = out;
    // ارقام فارسی/عربی -> لاتین
    t = t.replace(/[۰-۹٠-٩]/g, c => String((FA.indexOf(c) >= 0 ? FA : AR).indexOf(c)));
    t = t.replace(new RegExp('[' + DIACRITICS + ']', 'g'), '');
    // نقطه‌گذاری فارسی در همسایگی متن فارسی
    const FA_RANGE = '\\u0600-\\u06ff';
    t = t.replace(new RegExp('(?<=[' + FA_RANGE + '])\\?', 'g'), '؟')
         .replace(new RegExp('(?<=[' + FA_RANGE + ']);', 'g'), '؛')
         .replace(new RegExp('(?<=[' + FA_RANGE + ']),(?=\\s|$)', 'g'), '،')
         .replace(new RegExp('(?<=[' + FA_RANGE + ']),(?=[' + FA_RANGE + '])', 'g'), '،');
    t = t.replace(/[\u00a0\u2007\u202f]/g, ' ')
         .replace(/[ \t]{2,}/g, ' ')
         .replace(/\n{3,}/g, '\n\n')
         .replace(new RegExp('\\s+' + ZWNJ + '\\s+', 'g'), ' ')
         .trim();
    if (!keepZwnj) t = t.split(ZWNJ).join(' ');
    return t;
  }

  const WORD_RE = /[\u0600-\u06ff\u0750-\u077f0-9A-Za-z_\-]+/g;
  function tokenize(text) {
    return (normalize(text).toLowerCase().split(ZWNJ).join('').match(WORD_RE) || []);
  }

  function faNumWordToDigit(text) {
    return normalize(text, false).split(' ').map(p => D.fa_num_words[p] !== undefined ? String(D.fa_num_words[p]) : p).join(' ');
  }

  // ---------- TF-IDF با n-gram نویسه‌ای (char_wb, 3..5, sublinear) ----------
  const VOCAB = D.vocab;
  const IDF = D.idf || null;   // اگر داده قدیمی باشد، بدون IDF ادامه می‌دهیم

  function queryNgrams(text) {
    // دقیقاً هم‌راستا با sklearn char_wb: هر واژه با فاصله پد می‌شود و آن فاصله
    // *بخشی از* n-gram است (نشانگر مرز واژه). حذف این فاصله‌ها باعث واگرایی
    // امتیازها می‌شد (باگ گزارش‌شده: 0.2221 در پایتون در برابر 0.1974 در JS).
    const counts = new Map();
    for (const word of text.split(' ')) {
      const s = ' ' + word + ' ';
      for (let n = 3; n <= 5; n++) {
        for (let i = 0; i + n <= s.length; i++) {
          const g = s.slice(i, i + n);
          counts.set(g, (counts.get(g) || 0) + 1);
        }
      }
    }
    return counts;
  }

  function queryVector(text) {
    const counts = queryNgrams(text);
    const vec = [];
    let norm = 0;
    for (const [g, c] of counts) {
      const col = VOCAB[g];
      if (col === undefined) continue;
      // sklearn: (1 + log tf) * idf  — هر دو عامل لازم‌اند
      const w = (1 + Math.log(c)) * (IDF ? IDF[col] : 1);
      vec.push([col, w]);
      norm += w * w;
    }
    norm = Math.sqrt(norm) || 1e-9;
    return vec.map(([c, w]) => [c, w / norm]);
  }

  function cosine(qvec, docIdx) {
    const [cols, vals] = D.vectors[docIdx];
    let s = 0;
    for (const [c, w] of qvec) {
      const k = cols.indexOf(c);
      if (k >= 0) s += w * vals[k];
    }
    return s;
  }

  // ---------- تطبیق ماده ----------
  const ARTICLE_Q_RE = /(?:ماده|مادهٔ)\s*(?:\(\s*)?(\d+)(?:\s*\))?\s*(مکرر)?/g;
  const ARTICLE_C_RE = /ماده\s*(\d+)(\s*مکرر)?/g;

  function queryArticles(query) {
    const found = new Set();
    for (const variant of [query, faNumWordToDigit(query)]) {
      const t = normalize(variant, false);
      for (const m of t.matchAll(ARTICLE_Q_RE)) found.add(m[1] + (m[2] ? ' مکرر' : ''));
    }
    return found;
  }

  function chunkArticles(chunk) {
    const found = new Set();
    if (chunk.article) {
      const m = chunk.article.match(ARTICLE_C_RE);
      if (m) found.add(m[1] + (m[2] ? ' مکرر' : ''));
    }
    for (const m of chunk.text.slice(0, 120).matchAll(ARTICLE_C_RE)) found.add(m[1] + (m[2] ? ' مکرر' : ''));
    return found;
  }

  const STOP = new Set(D.stopwords);
  const INTENT_BOOST = {
    process: { 'فرآیند و سامانه': 1.35, 'خدمات': 1.10 },
    condition: { 'آیین‌نامه': 1.15, 'خدمات': 1.10 },
    amount: { 'آیین‌نامه': 1.15, 'خدمات': 1.15 },
    definition: { 'قانون': 1.15, 'اساسنامه': 1.10 },
    list: { 'خدمات': 1.20, 'توانمندسازی': 1.10 }
  };

  function search(query, topK = 5, minScore = 0.08, intent = null) {
    const qNorm = normalize(query);
    const qvec = queryVector(qNorm + ' ' + faNumWordToDigit(qNorm));
    const qTokens = new Set(tokenize(query).filter(t => !STOP.has(t) && t.length > 1));
    const qArticles = queryArticles(query);
    const hits = [];

    D.chunks.forEach((chunk, i) => {
      const tfidf = cosine(qvec, i);
      const body = new Set(chunk.tokens);
      let overlap = 0;
      for (const t of qTokens) if (body.has(t)) overlap++;
      const kw = qTokens.size ? overlap / qTokens.size : 0;
      const articleMatch = [...qArticles].some(a => chunkArticles(chunk).has(a));

      let score = 0.62 * tfidf + 0.22 * kw;
      if (articleMatch) score += 0.30;
      if (chunk.heading && normalize(query).includes(chunk.heading)) score += 0.05;
      if (intent && INTENT_BOOST[intent] && INTENT_BOOST[intent][chunk.category]) score *= INTENT_BOOST[intent][chunk.category];
      if (tfidf < 0.01 && kw === 0 && !articleMatch) return;

      hits.push({ chunk, score, tfidf, kw, articleMatch, ref: 0 });
    });

    hits.sort((a, b) => b.score - a.score);
    return hits.slice(0, topK).filter(h => h.score >= minScore);
  }

  function confidenceFromScore(s) { return s >= 0.45 ? 'high' : (s >= 0.22 ? 'medium' : 'low'); }

  // ---------- لایه تسهیلات (هم‌راستا با app/core/facilities.py) ----------
  const REG = D.facilities;

  function facilityList() { return REG.facilities || []; }

  function isFacilityQuery(q) {
    const t = normalize(q, false);
    return ['تسهیلات', 'وام', 'قرض', 'لیزینگ', 'سرمایه در گردش', 'نمونه‌سازی', 'نمونه سازی',
      'تولید صنعتی', 'ودیعه', 'رهن', 'اخزا', 'اوراق خزانه', 'اشتغال', 'کارآفرین', 'توانا',
      'خرید محل کار', 'محل کار', 'سقف', 'نرخ', 'بازپرداخت'].some(k => t.includes(k));
  }

  function wantsOverview(q) {
    const t = normalize(q, false);
    return ['انواع', 'چه تسهیلاتی', 'لیست', 'فهرست', 'همه', 'کدام', 'چند نوع', ' overview'].some(k => t.includes(k))
      && !facilityList().some(f => (f.aliases || []).some(a => normalize(a, false).length > 6 && t.includes(normalize(a, false))));
  }

  // واژه‌های پرکاربرد که «اشتراک» حساب نمی‌شوند (هم‌راستا با FACILITY_STOPWORDS پایتون)
  const FAC_STOP = new Set(['در','به','از','و','با','برای','تا','که','این','آن','یک','را','هم','چه','کجا','چگونه','چطور','آیا','است']);
  // واژه‌های عمومی دسته تسهیلات: هیچ تمایزی ایجاد نمی‌کنند (هم‌راستا با پایتون)
  const FAC_GENERIC = new Set(['تسهیلات','وام','قرض','قرض‌الحسنه','credit','loan','facility']);

  function matchFacilities(query, limit = 3) {
    const q = normalize(query, false);
    const qWords = new Set(q.match(/[\u0600-\u06ff]+/g) || []);
    const qFlat = q.split(' ').join('');
    const scored = [];
    for (const f of facilityList()) {
      let score = 0;
      for (const term of [f.name].concat(f.aliases || [])) {
        const tf = term.replace(/[\u200c\s]/g, '');
        if (tf && qFlat.includes(tf)) {
          if (FAC_GENERIC.has(tf)) continue;
          score += tf.length * 3;
          continue;
        }
        let shared = false;
        for (const w of term.split(/\s+/)) {
          if (FAC_STOP.has(w) || FAC_GENERIC.has(w)) continue;
          if (w.length >= 4 && qWords.has(normalize(w, false))) { score += w.length * 2; shared = true; }
        }
      }
      if (score) scored.push([score, f]);
    }
    if (!scored.length) return [];
    scored.sort((a, b) => b[0] - a[0] || a[1].name.localeCompare(b[1].name, 'fa'));
    const top = scored[0][0];
    const kept = top >= 10 ? scored.filter(s => s[0] >= top * 0.6) : scored;
    return kept.slice(0, limit).map(s => s[1]);
  }

  function voiceClean(v) {
    let out = String(v || '').replace(/[（(][^()（）]*[)）]/g, ' ').replace(/\s+/g, ' ').trim();
    for (const p of ['تا سقف', 'تا ']) if (out.startsWith(p)) { out = out.slice(p.length).trim(); break; }
    return out;
  }
  function voiceDuration(v) {
    let out = voiceClean(v);
    for (const p of ['دوره بازگشت حداکثر', 'دوره بازگشت', 'دوره بازپرداخت', 'بازپرداخت'])
      if (out.startsWith(p)) { out = out.slice(p.length).trim(); break; }
    return out && !out.startsWith('بازپرداخت') ? 'بازپرداخت ' + out : out;
  }
  function voiceRate(f) {
    for (const src of [f.rate, f.ceiling]) {
      const t = src || '';
      const num = t.match(/[\d۰-۹][\d۰-۹٬٫.]*\s*درصد/);
      if (num) {
        const tail = t.slice(num.index);
        const cut = ['؛', ';', '.'].map(s => tail.indexOf(s)).filter(i => i > 0);
        return voiceClean(tail.slice(0, cut.length ? Math.min(...cut) : tail.length));
      }
      const lab = t.match(/(?:نرخ|کارمزد)[^.؛;]*/);
      if (lab) { const c = voiceClean(lab[0]); if (c) return c; }
    }
    return '';
  }
  function facilityVoice(f) {
    const portal = (REG._meta.application_portal || 'ghazal.inif.ir').replace(/^https?:\/\//, '');
    const parts = [f.name + ': ' + voiceClean(f.ceiling)];
    const rate = voiceRate(f);
    if (rate) parts.push('با ' + (rate.startsWith('با ') ? rate.slice(3).trim() : rate));
    if (f.duration) parts.push(voiceDuration(f.duration));
    if (f.purpose) {
      let p = voiceClean(f.purpose).split('؛')[0];
      if (p.length > 90) p = p.slice(0, 90).split('،').slice(0, -1).join('،');
      parts.push('برای ' + p);
    }
    let v = parts.join('، ') + '.';
    const tail = ' درخواست خود را در ' + portal + ' ثبت کنید.';
    if (v.length + tail.length <= 380) v += tail;
    return v;
  }

  function facilityCard(f, index) {
    const rows = [['سقف', f.ceiling], ['نرخ', f.rate], ['مدت', f.duration], ['کاربرد', f.purpose], ['شرایط', f.conditions]];
    const body = rows.filter(r => r[1]).map(r => r[0] + ': ' + r[1]).join('\n');
    const legal = f.legal_basis ? 'مبنای قانونی: ' + f.legal_basis + ' | ' : '';
    return (index ? index + ') ' : '') + f.name + '\n' + '─'.repeat(20) + '\n' + body +
      '\n' + legal + 'منبع: ' + f.source + ' (' + f.source_date + ')';
  }

  function facilityTable(items) {
    const rows = items.map(f => [f.name, f.ceiling, f.rate, f.duration].map(v => v || '—').join(' | '));
    return 'خدمت | سقف | نرخ | مدت\n' + '─'.repeat(20) + '\n' + rows.join('\n');
  }

  // ---------- ساخت پاسخ ----------
  const LEAD = {
    amount: 'بر پایه اسناد صندوق، مقادیر پرسش‌شده به این شرح است:',
    condition: 'شرایط ذکرشده در اسناد صندوق به این صورت است:',
    process: 'مراحل و مسیر اقدام طبق اسناد صندوق:',
    definition: 'تعریف و جایگاه قانونی مورد پرسش:',
    list: 'موارد خواسته‌شده در اسناد صندوق:',
    default: 'بر پایه اسناد رسمی صندوق:'
  };
  const TEMPLATE = '{lead}\n\n{evidence}\n\n📚 منابع:\n{sources}';

  function detectIntent(q) {
    const t = normalize(q, false);
    if (/چقدر|سقف|مبلغ|میزان|نرخ|درصد|چند/.test(t)) return 'amount';
    if (/شرایط|الزام|باید|نیاز|وثیقه|ضمانت/.test(t)) return 'condition';
    if (/چگونه|چطور|مراحل|از کجا|روش|سامانه/.test(t)) return 'process';
    if (/یعنی|تعریف|چیست|ماهو|چه می‌گوید/.test(t)) return 'definition';
    if (/انواع|لیست|فهرست|کدام|موارد/.test(t)) return 'list';
    return 'default';
  }

  function trimEvidence(text, limit = 320) {
    let t = normalize(text)
      .replace(/^ماده\s+\d+(?:\s*مکرر)?\s*[-–—:]\s*/, '')
      .replace(/^#+\s*/, '')
      .replace(/^\s*پرسش\s*[:：][^\n]{0,160}?\s*(?:\؟|\?)\s*(?:پاسخ\s*[:：])?\s*/, '')
      .replace(/^\s*پاسخ\s*[:：]\s*/, '')
      .replace(/\s+/g, ' ').trim();
    if (t.length <= limit) return t;
    const cut = t.slice(0, limit);
    return cut.slice(0, Math.max(cut.lastIndexOf('.'), cut.lastIndexOf('،'))) + '…';
  }

  function firstSentence(text, limit = 170) {
    const t = normalize(text).trim();
    for (const sep of ['، ', '. ', '؛ ']) {
      const i = t.indexOf(sep);
      if (i >= 40 && i <= limit) return t.slice(0, i).replace(/[،؛,. ]+$/, '');
    }
    return t.slice(0, limit).replace(/[،؛,. ]+$/, '');
  }

  function uniqueSources(hits) {
    const sources = [], refMap = {};
    for (const h of hits) {
      if (refMap[h.chunk.doc] === undefined) {
        refMap[h.chunk.doc] = sources.length + 1;
        sources.push({ doc_id: h.chunk.doc, title: h.chunk.title, category: h.chunk.category, source: h.chunk.source });
      }
      h.ref = refMap[h.chunk.doc];
    }
    return { sources, refMap };
  }

  function answer(query) {
    const started = performance.now();
    const intent = detectIntent(query);
    const hits = search(query, 5, 0.08, intent);
    const result = {
      query, intent, confidence: 'low', top_score: 0, citations: [], sources: [],
      followups: [], mode: 'extractive', facilities: [], registry_date: '', latency_ms: 0
    };

    // رجیستری تسهیلات مستقل از بازیابی معنایی است؛ پس پیش از هر بازگشت زودهنگام
    // بررسی می‌شود (هم‌راستا با rag.py).
    const facilityMatch = isFacilityQuery(query) && (!wantsOverview(query) ? matchFacilities(query) : []);

    if (!hits.length && !facilityMatch.length && !wantsOverview(query)) {
      result.mode = 'no_evidence';
      result.answer = 'در اسناد بارگذاری‌شده، شاهد معتبری برای این پرسش پیدا نکردم.\n' +
        'برای پرهیز از حدس‌زدن پاسخی نمی‌دهم. می‌توانید در سامانه غزال (ghazal.inif.ir) پیگیری کنید.';
      result.voice_text = 'در اسناد موجود پاسخی برای این پرسش پیدا نشد. لطفا در سامانه غزال پیگیری کنید.';
      result.latency_ms = Math.round(performance.now() - started);
      return result;
    }

    result.top_score = +hits[0].score.toFixed(4);
    result.confidence = confidenceFromScore(hits[0].score);

    // فیلتر ربط + حذف تکرار (هم‌راستا با سرور)
    let kept = hits;
    if (hits.length > 2) {
      const cutoff = 0.5 * hits[0].score;
      const f = hits.filter(h => h.score >= cutoff);
      kept = f.length >= 2 ? f : hits.slice(0, 2);
    }
    const seen = {};
    kept = kept.filter(h => {
      const k = h.chunk.doc + '|' + h.chunk.article + '|' + h.chunk.heading;
      seen[k] = (seen[k] || 0) + 1;
      return !(seen[k] > 2 && kept.length >= 2);
    });
    kept = kept.slice(0, 4);

    const { sources } = uniqueSources(kept);
    result.sources = sources;
    result.citations = kept.map((h, i) => ({ index: i + 1, ref: h.ref, citation: h.chunk.citation, article: h.chunk.article }));

    // --- حالت تسهیلات ---
    let facilityAnswer = null;
    if (isFacilityQuery(query) && (facilityMatch.length || wantsOverview(query))) {
      const portal = REG._meta.application_portal || 'https://ghazal.inif.ir';
      const date = REG._meta.last_reviewed || '';
      const warn = '⚠️ نرخ‌ها و سقف‌ها بر پایه فراخوان‌های صندوق به‌روزرسانی می‌شوند؛ پیش از اقدام، شرایط روز را در سامانه غزال بررسی کنید.';
      if (wantsOverview(query)) {
        const items = facilityList();
        facilityAnswer = {
          overview: true,
          text: '📋 تسهیلات صندوق نوآوری و شکوفایی (نمای کلی)\n\n' + facilityTable(items) +
            '\n\n🔗 ثبت همه درخواست‌ها: ' + portal + '\n🕓 آخرین بازبینی ارقام: ' + date + '\n' + warn,
          voice: 'صندوق ' + items.length + ' نوع تسهیلات دارد، از نمونه‌سازی قرض‌الحسنه با نرخ چهار درصد تا تولید صنعتی با نرخ یازده درصد و دوره بازگشت هفت ساله. ثبت درخواست در سامانه غزال انجام می‌شود.',
          matched: items.map(f => ({ id: f.id, name: f.name }))
        };
      } else {
        const matched = facilityMatch;
        if (matched.length) {
          const blocks = matched.map((f, i) => facilityCard(f, matched.length > 1 ? i + 1 : null));
          facilityAnswer = {
            overview: false,
            text: '📊 تسهیلات صندوق نوآوری و شکوفایی\n\n' + blocks.join('\n\n') +
              '\n\n🔗 ثبت درخواست: ' + portal + '\n🕓 آخرین بازبینی ارقام: ' + date + '\n' + warn,
            voice: facilityVoice(matched[0]),
            matched
          };
        }
      }
    }

    if (facilityAnswer) {
      result.mode = 'facilities';
      result.confidence = 'high';
      result.facilities = facilityAnswer.matched;
      result.registry_date = REG._meta.last_reviewed || '';
      const labels = [];
      for (const h of kept.slice(0, 2)) {
        const l = '[' + h.ref + '] ' + h.chunk.citation;
        if (!labels.includes(l)) labels.push(l);
      }
      result.answer = facilityAnswer.text + (labels.length ? '\n\n📎 مستندات قانونی مرتبط:\n' + labels.map(l => '• ' + l).join('\n') : '');
      result.voice_text = facilityAnswer.voice;
    } else if (result.confidence === 'low') {
      result.mode = 'low_confidence';
      result.answer = 'شاهد کافی در اسناد برای پاسخ قطعی پیدا نشد (امتیاز ' + result.top_score + ').\n' +
        'نزدیک‌ترین مطالب:\n' + kept.slice(0, 2).map(h => '• ' + trimEvidence(h.chunk.text, 160) + '\n   — ' + h.chunk.citation).join('\n') +
        '\n\nبرای پاسخ قطعی با کارشناسان صندوق از طریق سامانه غزال تماس بگیرید.';
      result.voice_text = 'شاهد کافی در اسناد پیدا نشد. لطفا از سامانه غزال پیگیری کنید.';
    } else {
      const lead = LEAD[intent] || LEAD.default;
      const evidence = kept.map(h => '• ' + trimEvidence(h.chunk.text) + '\n   — ' + h.chunk.citation + (h.ref ? ' [' + h.ref + ']' : '')).join('\n');
      const listed = {};
      for (const h of kept) if (!listed[h.ref]) listed[h.ref] = h.chunk.title + ' | ' + h.chunk.source;
      const srcLines = Object.keys(listed).sort((a, b) => a - b).map(i => '[' + i + '] ' + listed[i]).join('\n');
      result.answer = TEMPLATE.replace('{lead}', lead).replace('{evidence}', evidence).replace('{sources}', srcLines).trim();
      const top = kept[0];
      const shortCite = top.chunk.article ? top.chunk.title + ' - ' + top.chunk.article : top.chunk.title;
      result.voice_text = lead + ' به استناد ' + shortCite + ': ' + firstSentence(trimEvidence(top.chunk.text, 200), 140) + '. جزئیات در متن پاسخ است.';
    }

    result.followups = ['انواع تسهیلات صندوق چیست؟', 'برای دریافت تسهیلات از کجا اقدام کنم؟', 'ضمانت‌نامه اعتباری تا چه سقفی صادر می‌شود؟']
      .filter(f => f !== query).slice(0, 3);
    result.latency_ms = Math.round(performance.now() - started);
    return result;
  }

  // ---------- مهارت‌های محاسباتی/تحلیلی (خارج از answer() تا هم‌ترازی دست‌نخورده بماند) ----------

  // تبدیل هر عددی (فارسی/عربی/کلمه‌ای) به عدد JS؛ برای ماشین‌حساب
  function toNumber(text) {
    const t = faNumWordToDigit(String(text || ''));
    const m = t.replace(/[٬،,\s]/g, '').match(/\d+(?:\.\d+)?/);
    return m ? parseFloat(m[0]) : null;
  }

  // قسط ماهانه به روش آنوئیتی (اصل+سود یکسان). ورودی: مبلغ کل، درصد سود سالانه، تعداد ماه
  function installment(principal, annualPercent, months) {
    const P = principal, n = months;
    const r = annualPercent / 100 / 12;
    const monthly = r === 0 ? P / n : (P * r) / (1 - Math.pow(1 + r, -n));
    const total = monthly * n;
    const fa = x => Math.round(x).toLocaleString('fa-IR');
    return {
      monthly, total, interest: total - P,
      text: 'قسط ماهانه: حدود ' + fa(monthly) + ' ریال\nمجموع بازپرداخت: حدود ' + fa(total) +
        ' ریال\nسود پرداختی: حدود ' + fa(total - P) + ' ریال\n(محاسبه تقریبی به روش آنوئیتی؛ ملاک، اعلام بانک عامل است.)'
    };
  }

  // جدول مقایسهٔ چند تسهیلات بر پایهٔ رجیستری
  function compareFacilities(facs) {
    if (!facs || facs.length < 2) return null;
    const row = (label, get) => label + ': ' + facs.map(f => get(f) || '—').join(' | ');
    const lines = ['مقایسهٔ ' + facs.map(f => f.name).join(' | ')];
    lines.push(row('سقف', f => f.ceiling));
    lines.push(row('نرخ', f => f.rate));
    lines.push(row('مدت', f => f.duration));
    lines.push(row('مبنای قانونی', f => f.legal_basis || 'فراخوان صندوق'));
    return lines.join('\n');
  }

  // فهرست اسناد نمایه‌شده
  function docList() {
    const byDoc = {};
    for (const c of D.chunks) (byDoc[c.doc] = byDoc[c.doc] || { title: c.title, source: c.source, chunks: 0 }).chunks++;
    return Object.values(byDoc);
  }

  // یافتن متن دقیق یک ماده
  function findArticle(num) {
    const target = String(parseInt(num, 10));
    const hits = D.chunks.filter(c => c.article && /ماده\s*(\d+)/.test(c.article) &&
      (/ماده\s*(\d+)/.exec(c.article))[1] === target);
    return hits;
  }

  // دروازهٔ مهارت‌ها: اگر پرسش یکی از مهارت‌ها را فعال کرد، پاسخ آماده برمی‌گردد؛
  // وگرنه null تا به answer() عادی برود. هم وب و هم بات‌ها از همین تابع استفاده می‌کنند.
  function skillRoute(query) {
    const q = String(query || '').trim();

    // ۱) ماشین‌حساب اقساط: «اگر ۱۰ میلیارد با سود ۱۱٪ برای ۵ سال قسط‌بندی شود…»
    if (/قسط|اقساط|ماشین.?حساب|محاسبه/.test(q)) {
      const unitMult = w => w === 'میلیارد' ? 1e9 : w === 'میلیون' ? 1e6 : w === 'هزار' ? 1e3 : 1;
      const amt = q.match(/([۰-۹٠-٩0-9][,٬\s۰-۹٠-٩0-9]*)\s*(میلیارد|میلیون|هزار)?\s*(ریال|تومان)?/);
      const pctM = q.match(/([۰-۹٠-٩0-9]+)\s*(?:درصد|٪|%)/);
      const salM = q.match(/([۰-۹٠-٩0-9]+)\s*سال/);
      const mahM = q.match(/([۰-۹٠-٩0-9]+)\s*ماه/);
      if (amt && amt[1] && pctM) {
        let P = toNumber(amt[1]) * unitMult(amt[2]);
        if (amt[3] === 'تومان') P *= 10;   // تبدیل تومان به ریال
        const pct = toNumber(pctM[1]);
        const n = mahM ? toNumber(mahM[1]) : (salM ? toNumber(salM[1]) * 12 : 12);
        const r = installment(P, pct, n);
        return { mode: 'calculator', confidence: 'high', top_score: 1, citations: [], sources: [],
          followups: [], answer: r.text,
          voice_text: 'قسط ماهانه حدوداً ' + Math.round(r.monthly).toLocaleString('fa-IR') + ' ریال است.' };
      }
    }

    // ۲) مقایسهٔ تسهیلات
    if (/مقایسه|فرق|تفاوت|کدام.{0,12}(بهتر|مناسب)/.test(q)) {
      const facs = matchFacilities(q, 4);
      const cmp = compareFacilities(facs);
      if (cmp) return { mode: 'compare', confidence: 'high', top_score: 1, citations: [],
        sources: [], followups: [], answer: cmp, voice_text: 'جدول مقایسه در پاسخ متنی آمده است.' };
    }

    // ۳) فهرست اسناد
    if (/چه اسناد|کدام سند|فهرست اسناد|لیست اسناد|چه مدارکی.{0,6}نمایه/.test(q)) {
      const docs = docList();
      return { mode: 'docs', confidence: 'high', top_score: 1, citations: [], sources: [],
        followups: [], answer: 'اسناد نمایه‌شده (' + docs.length + '):\n' +
          docs.map(d => '• ' + d.title + ' — ' + d.chunks + ' بخش').join('\n'),
        voice_text: 'این دستیار از ' + docs.length.toLocaleString('fa-IR') + ' سند رسمی پاسخ می‌دهد.' };
    }

    // ۴) متن دقیق ماده
    const am = q.match(/ماده\s*([۰-۹٠-٩0-9]+)\s*(را|چیست|بگو|نشان|متن)?\s*$/);
    if (am) {
      const hits = findArticle(faNumWordToDigit(am[1]));
      if (hits.length) {
        return { mode: 'article', confidence: 'high', top_score: 1, citations: [],
          sources: hits.map(h => ({ title: h.title, source: h.source })), followups: [],
          answer: hits.map(h => '• ' + h.text + '\n   — ' + h.citation).join('\n\n'),
          voice_text: 'متن مادهٔ ' + faNumWordToDigit(am[1]) + ' در پاسخ آمده است.' };
      }
    }

    return null;
  }

  global.INIF_CORE = {
    normalize, tokenize, faNumWordToDigit, search, answer, isFacilityQuery, wantsOverview, matchFacilities,
    facilityList, skillRoute, installment, compareFacilities, docList, findArticle,
    stats: () => ({ chunks: D.chunks.length, documents: new Set(D.chunks.map(c => c.doc)).size, vocabulary: Object.keys(VOCAB).length })
  };
})(window);
