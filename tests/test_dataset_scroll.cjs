// Exercise the actual component script with deterministic layout and input events.
const {readFileSync} = require('node:fs');
const {runInNewContext} = require('node:vm');
const assert = require('node:assert/strict');
const source = readFileSync('utils/dataset_workspace.py', 'utf8');
const script = source.match(/js='''\r?\n([\s\S]*?)\r?\n'''/)[1];

function mount() {
    let now = 0, next = 0, scrolls = 0;
    const timers = new Map(), listeners = new Map(), observers = [];
    const scroller = {style: {overflowAnchor: 'auto'}};
    const heading = {style: {scrollMarginTop: ''}, parentElement: scroller,
        isConnected: true, scrollIntoView: () => scrolls++};
    const content = {};
    const view = {querySelector: () => heading, closest: () => content};
    class Observer {
        constructor(callback) {this.callback = callback; observers.push(this);}
        observe() {this.active = true;}
        disconnect() {this.active = false;}
    }
    const cleanup = runInNewContext(script.replace('export default function', '(function') + ')', {
        document: {scrollingElement: scroller,
            addEventListener: (event, fn) => listeners.set(event, fn),
            removeEventListener: event => listeners.delete(event)},
        getComputedStyle: () => ({overflowY: 'auto'}),
        ResizeObserver: Observer, MutationObserver: Observer,
        setTimeout: (fn, delay) => {timers.set(++next, {fn, at: now + delay}); return next;},
        clearTimeout: id => timers.delete(id),
    })({parentElement: {closest: () => view}});
    return {
        tick(ms) {
            const end = now + ms;
            while (true) {
                const due = [...timers].filter(([, t]) => t.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
                if (!due) break;
                now = due[1].at; timers.delete(due[0]); due[1].fn();
            }
            now = end;
        },
        layout() {observers.filter(o => o.active).forEach(o => o.callback());},
        input(event) {listeners.get(event)?.();},
        get scrolls() {return scrolls;}, scroller, heading, cleanup, timers, listeners,
    };
}

const page = mount();
assert.equal(page.scroller.style.overflowAnchor, 'none');
page.tick(150);
assert.equal(page.scrolls, 1);
page.tick(500); page.layout(); page.tick(150);
assert.equal(page.scrolls, 2, 'Late dataset/footer layout must trigger another alignment');
page.tick(2000);
assert.equal(page.scroller.style.overflowAnchor, 'auto');
assert.equal(page.listeners.size, 0);
const finalScrolls = page.scrolls;
page.layout(); page.tick(500);
assert.equal(page.scrolls, finalScrolls, 'No corrections after the navigation window');

for (const event of ['wheel', 'touchstart', 'pointerdown', 'keydown']) {
    const page = mount();
    page.input(event); page.layout(); page.tick(3000);
    assert.equal(page.scrolls, 0, `${event} must cancel pending corrections`);
    assert.equal(page.scroller.style.overflowAnchor, 'auto');
    assert.equal(page.heading.style.scrollMarginTop, '');
}
const removed = mount();
removed.cleanup(); removed.tick(3000);
assert.equal(removed.scrolls, 0);
assert.equal(removed.timers.size, 0);
console.log('Dataset scroll timing, late layout, user cancellation, and cleanup checks passed.');
