const header = document.querySelector('[data-header]');
const menuToggle = document.querySelector('[data-menu-toggle]');
const mobileNav = document.querySelector('[data-mobile-nav]');

window.addEventListener('scroll', () => {
  header.classList.toggle('scrolled', window.scrollY > 12);
});

menuToggle?.addEventListener('click', () => {
  const isOpen = mobileNav.classList.toggle('open');
  menuToggle.setAttribute('aria-expanded', String(isOpen));
});

mobileNav?.querySelectorAll('a').forEach((link) => {
  link.addEventListener('click', () => {
    mobileNav.classList.remove('open');
    menuToggle?.setAttribute('aria-expanded', 'false');
  });
});

const quickstartCommands = {
  train: {
    file: 'property_prediction / train.py',
    command: 'python property_prediction/train.py -c property_prediction/configs/megnet/megnet_mp2018_train_60k_e_form.yaml',
    html: '<span class="token-purple">python</span> property_prediction/train.py -c property_prediction/configs/megnet/megnet_mp2018_train_60k_e_form.yaml',
    status: 'training entry <span class="mono">·</span> GPU / CPU',
  },
  predict: {
    file: 'property_prediction / predict.py',
    command: "python property_prediction/predict.py --model_name='megnet_mp2018_train_60k_e_form' --weights_name='best.pdparams' --cif_file_path='./property_prediction/example_data/cifs/' --save_path='result.csv'",
    html: String.raw`<span class="token-purple">python</span> property_prediction/predict.py <span class="token-blue">--model_name='megnet_mp2018_train_60k_e_form'</span> <span class="token-blue">--weights_name='best.pdparams'</span> <span class="token-blue">--cif_file_path='./property_prediction/example_data/cifs/'</span> <span class="token-blue">--save_path='result.csv'</span>`,
    status: 'prediction output <span class="mono">·</span> CSV',
  },
};

const codeOutput = document.querySelector('[data-code-output]');
const codeFile = document.querySelector('[data-code-file]');
const codeStatus = document.querySelector('[data-code-status]');
const codeCopy = document.querySelector('.code-copy');

document.querySelectorAll('[data-code-tab]').forEach((tab) => {
  tab.addEventListener('click', () => {
    const mode = tab.getAttribute('data-code-tab');
    const command = quickstartCommands[mode];
    if (!command || !codeOutput || !codeFile || !codeStatus || !codeCopy) return;

    document.querySelectorAll('[data-code-tab]').forEach((item) => {
      const active = item === tab;
      item.classList.toggle('active', active);
      item.setAttribute('aria-selected', String(active));
    });

    codeFile.textContent = command.file;
    codeOutput.innerHTML = command.html;
    codeStatus.innerHTML = `<span class="status-dot"></span> ${command.status}`;
    codeCopy.setAttribute('data-copy', command.command);
  });
});

document.querySelectorAll('[data-copy]').forEach((button) => {
  button.addEventListener('click', async () => {
    const target = button.getAttribute('data-copy');
    const text = target || quickstartCommands.train.command;
    try {
      await navigator.clipboard.writeText(text);
      const original = button.textContent;
      button.textContent = '已复制 ✓';
      setTimeout(() => { button.textContent = original; }, 1600);
    } catch {
      button.textContent = '请手动复制';
    }
  });
});

const revealItems = document.querySelectorAll('.capability-card, .flow-step, .code-card, .install-card');
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.style.animation = 'rise-in .65s cubic-bezier(.2,.8,.2,1) both';
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.14 });
revealItems.forEach((item) => observer.observe(item));
