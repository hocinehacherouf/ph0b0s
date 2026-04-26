// Intentionally bad sample for the ph0b0s integration test.
// The mocked LLM-toy detector will flag the hardcoded credential; the
// faked cargo-audit subprocess will emit a canned RUSTSEC advisory.
fn main() {
    let pwd = "hunter2"; // hardcoded password
    println!("password is {}", pwd);
}
