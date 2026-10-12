//! Global allocator: musl's malloc serialises every thread on one lock, so the two search threads would queue
//! on the many small vectors of a candidate evaluation. A thread that turns its cache on keeps up to KEEP freed
//! blocks per power-of-two size class (16 B .. 4 KiB) and hands them out again; everything else, and every other
//! thread, uses the system allocator. A worker turns its cache off (returning the blocks) before it ends.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::UnsafeCell;
use std::ptr::null_mut;

const CLASSES: usize = 9;
const KEEP: usize = 128;

struct Cache {
    on: bool,
    n: [usize; CLASSES],
    free: [[*mut u8; KEEP]; CLASSES],
}

thread_local! {
    static CACHE: UnsafeCell<Cache> = const { UnsafeCell::new(Cache { on: false, n: [0; CLASSES], free: [[null_mut(); KEEP]; CLASSES] }) };
}

/// Size class of a layout (blocks of 16 << c bytes, 16-aligned), or None for the system allocator.
fn class(l: &Layout) -> Option<usize> {
    if l.align() > 16 || l.size() > 16 << (CLASSES - 1) {
        return None;
    }
    Some(l.size().max(16).next_power_of_two().trailing_zeros() as usize - 4)
}

fn block(c: usize) -> Layout {
    // SAFETY: 16 << c is a non-zero multiple of the power-of-two alignment 16, far below isize::MAX.
    unsafe { Layout::from_size_align_unchecked(16 << c, 16) }
}

/// Run f on this thread's cache (None when the thread has it off or its storage is gone).
fn with<R>(f: impl FnOnce(&mut Cache) -> R) -> Option<R> {
    // SAFETY: the cache belongs to this thread, and f never allocates or re-enters it.
    CACHE.try_with(|k| unsafe { &mut *k.get() }).ok().filter(|k| k.on).map(f)
}

pub struct Alloc;

unsafe impl GlobalAlloc for Alloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let Some(c) = class(&l) else { return System.alloc(l) };
        let kept = with(|k| (k.n[c] > 0).then(|| {
            k.n[c] -= 1;
            k.free[c][k.n[c]]
        }));
        kept.flatten().unwrap_or_else(|| System.alloc(block(c)))
    }

    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        if class(&l).is_none() {
            return System.alloc_zeroed(l);
        }
        let p = self.alloc(l);
        if !p.is_null() {
            std::ptr::write_bytes(p, 0, l.size());
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        let Some(c) = class(&l) else { return System.dealloc(p, l) };
        let kept = with(|k| k.n[c] < KEEP && {
            k.free[c][k.n[c]] = p;
            k.n[c] += 1;
            true
        });
        if kept != Some(true) {
            System.dealloc(p, block(c));
        }
    }

    unsafe fn realloc(&self, p: *mut u8, l: Layout, size: usize) -> *mut u8 {
        let new = Layout::from_size_align_unchecked(size, l.align());
        match (class(&l), class(&new)) {
            (Some(a), Some(b)) if a == b => p,
            (None, None) => System.realloc(p, l, size),
            _ => {
                let q = self.alloc(new);
                if !q.is_null() {
                    std::ptr::copy_nonoverlapping(p, q, l.size().min(size));
                    self.dealloc(p, l);
                }
                q
            }
        }
    }
}

/// Turn this thread's cache on or off (off returns its blocks to the system allocator).
pub fn cache(on: bool) {
    let _ = CACHE.try_with(|k| {
        // SAFETY: as in `with`; System.dealloc does not use the cache.
        let k = unsafe { &mut *k.get() };
        for c in 0..CLASSES {
            for &p in &k.free[c][..k.n[c]] {
                unsafe { System.dealloc(p, block(c)) };
            }
            k.n[c] = 0;
        }
        k.on = on;
    });
}
